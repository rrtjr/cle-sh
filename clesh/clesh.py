import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from loguru import logger
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")


class CLESH:
    """
    Comprehensive Literal Explanation package for SHapley values by statistical validity.

    A statistical framework for interpreting SHAP values through rigorous significance testing,
    univariate pattern analysis, and feature interaction detection. Based on methodology from
    arxiv.org/abs/2409.12578.

    Parameters
    ----------
    alpha : float, default=0.05
        Statistical significance threshold for feature selection tests.
    p_univariate : float, default=0.05
        P-value threshold for univariate pattern significance.
    p_interaction : float, default=0.05
        P-value threshold for interaction significance.
    candidate_num_min : int, default=10
        Minimum number of candidate features to consider.
    candidate_num_max : int, default=20
        Maximum number of candidate features to consider.
    interaction_threshold : float, default=0.01
        Legacy threshold parameter, retained for compatibility.
    max_interactions_per_feature : int, default=3
        Maximum number of interactions to analyze per feature.
    normalize_interactions : bool, default=True
        Whether to normalize interaction strength relative to prediction range.
        Set to False to maintain strict absolute SHAP strength values.
    log_level : str, default="INFO"
        Logging level for analysis progress. Options: "DEBUG", "INFO", "WARNING", "ERROR".
    log_format : str, optional
        Custom log format string. If None, uses default loguru format.

    Attributes
    ----------
    results : dict
        Dictionary containing all analysis results including significance analysis,
        pattern analysis, feature types, and interaction analysis.
    """

    def __init__(
        self,
        alpha=0.05,
        p_univariate=0.05,
        p_interaction=0.05,
        candidate_num_min=10,
        candidate_num_max=20,
        interaction_threshold=0.01,
        max_interactions_per_feature=3,
        normalize_interactions=True,
        log_level="INFO",
        log_format=None,
    ):
        self.alpha = alpha
        self.p_univariate = p_univariate
        self.p_interaction = p_interaction
        self.candidate_num_min = candidate_num_min
        self.candidate_num_max = candidate_num_max
        self.interaction_threshold = interaction_threshold
        self.max_interactions_per_feature = max_interactions_per_feature
        self.normalize_interactions = normalize_interactions
        self.results = {}

        # Configure logger
        self._setup_logger(log_level, log_format)

    def _setup_logger(self, log_level, log_format):
        """
        Configure loguru logger for CLESH analysis.

        Parameters
        ----------
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format : str, optional
            Custom format string for log messages.
        """
        # Remove default handler to avoid duplication
        logger.remove()

        # Set default format if none provided
        if log_format is None:
            log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>CLE-SH</cyan> | <level>{message}</level>"

        # Add console handler with specified level and format
        logger.add(
            lambda msg: print(msg, end=""),  # Print to console without extra newline
            format=log_format,
            level=log_level,
            colorize=True,
        )

        self.logger = logger

    def _get_feature_type(self, x_col):
        """
        Classify feature type based on number of unique values.

        Parameters
        ----------
        x_col : array-like
            Feature values to classify.

        Returns
        -------
        str
            Feature type: 'binary', 'discrete', or 'continuous'.
        """
        unique_vals = len(np.unique(x_col))
        if unique_vals <= 2:
            return "binary"
        elif unique_vals <= 20:
            return "discrete"
        else:
            return "continuous"

    def analyze_shap_significance(self, shap_values, X, feature_names=None):
        """
        Identify statistically significant features using ranked SHAP value testing.

        Ranks features by descending mean absolute SHAP values and performs paired
        statistical tests between adjacent features to determine significance cutoff.
        Uses t-tests for normal distributions and Wilcoxon tests otherwise.

        Parameters
        ----------
        shap_values : array-like of shape (n_samples, n_features)
            SHAP values for each sample and feature.
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        feature_names : list of str, optional
            Names of features. If None, generates default names.

        Returns
        -------
        list of str
            Names of statistically significant features.
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]

        abs_shap = np.abs(shap_values)
        mean_abs_shap = np.mean(abs_shap, axis=0)

        indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_mean_abs = mean_abs_shap[indices]

        diffs = []
        p_values = []
        for i in range(len(sorted_features) - 1):
            group1 = abs_shap[:, indices[i]]
            group2 = abs_shap[:, indices[i + 1]]
            diff = sorted_mean_abs[i] - sorted_mean_abs[i + 1]
            diffs.append(diff)

            _, p_shap1 = stats.shapiro(group1)
            _, p_shap2 = stats.shapiro(group2)

            if p_shap1 > 0.05 and p_shap2 > 0.05:
                t_stat, p_val = stats.ttest_rel(group1, group2, alternative="greater")
            else:
                try:
                    t_stat, p_val = stats.wilcoxon(group1 - group2, alternative="greater")
                except ValueError:
                    # Handle case where all differences are zero (perfectly correlated features)
                    p_val = 1.0  # No significant difference

            p_values.append(p_val)

        candidate_ks = [i + 1 for i, p in enumerate(p_values) if p < self.alpha]
        filtered_ks = [
            k for k in candidate_ks if self.candidate_num_min <= k <= self.candidate_num_max
        ]

        self.logger.debug(f"Found {len(candidate_ks)} candidate cuts with p < {self.alpha}")
        self.logger.debug(
            f"Filtered to {len(filtered_ks)} cuts within range [{self.candidate_num_min}, {self.candidate_num_max}]"
        )

        if not filtered_ks:
            if candidate_ks:
                k = max(min(max(candidate_ks), self.candidate_num_max), self.candidate_num_min)
                self.logger.debug(f"No valid cuts in range, using fallback k={k}")
            else:
                k = self.candidate_num_min
                self.logger.debug(f"No significant cuts found, using minimum k={k}")
        else:
            filtered_indices = [candidate_ks.index(k) for k in filtered_ks]
            best_idx = np.argmax([diffs[i] for i in filtered_indices])
            k = filtered_ks[best_idx]
            self.logger.debug(f"Selected k={k} with highest difference among valid cuts")

        significant_features = sorted_features[:k]

        significance_results = {
            f: {"mean_abs_shap": mean_abs_shap[i]} for i, f in enumerate(feature_names)
        }
        self.results["significance_analysis"] = significance_results
        self.results["significant_features"] = significant_features

        return significant_features

    def _fit_linear(self, x, y):
        """
        Fit linear model and test coefficient significance.

        Parameters
        ----------
        x : array-like
            Input variable.
        y : array-like
            Target variable (SHAP values).

        Returns
        -------
        tuple or (None, None)
            (rmse, fit_info) if significant, (None, None) otherwise.
        """
        X_sm = sm.add_constant(x)
        model = sm.OLS(y, X_sm).fit()
        p_a = model.pvalues[1]
        if p_a >= self.p_univariate:
            return None, None
        y_pred = model.predict(X_sm)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        slope = model.params[1]
        intercept = model.params[0]
        return rmse, {"type": "linear", "slope": slope, "intercept": intercept, "p_a": p_a}

    def _fit_quadratic(self, x, y):
        """
        Fit quadratic model and test coefficient significance.

        Parameters
        ----------
        x : array-like
            Input variable.
        y : array-like
            Target variable (SHAP values).

        Returns
        -------
        tuple or (None, None)
            (rmse, fit_info) if significant, (None, None) otherwise.
        """
        df = pd.DataFrame({"x": x, "x2": x**2})
        X_sm = sm.add_constant(df)
        model = sm.OLS(y, X_sm).fit()
        p_a = model.pvalues["x2"]
        if p_a >= self.p_univariate:
            return None, None
        y_pred = model.predict(X_sm)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        a = model.params["x2"]
        b = model.params["x"]
        c = model.params["const"]
        return rmse, {"type": "quadratic", "a": a, "b": b, "c": c, "p_a": p_a}

    def _fit_sigmoid(self, x, y):
        """
        Fit sigmoid model and test coefficient significance.

        Parameters
        ----------
        x : array-like
            Input variable.
        y : array-like
            Target variable (SHAP values).

        Returns
        -------
        tuple or (None, None)
            (rmse, fit_info) if significant, (None, None) otherwise.
        """

        def sigmoid_func(x, L, a, x0, b):
            return L / (1 + np.exp(-a * (x - x0))) + b

        try:
            popt, pcov = curve_fit(
                sigmoid_func,
                x,
                y,
                p0=[np.max(y) - np.min(y), 1, np.mean(x), np.min(y)],
                maxfev=10000,
            )
            stds = np.sqrt(np.diag(pcov))
            a, std_a = popt[1], stds[1]
            df = len(x) - len(popt)
            t_val = a / std_a
            p_a = 2 * t_dist.sf(np.abs(t_val), df)
            if p_a >= self.p_univariate:
                return None, None
            y_pred = sigmoid_func(x, *popt)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            L, a, x0, b = popt
            return rmse, {"type": "sigmoid", "L": L, "a": a, "x0": x0, "b": b, "p_a": p_a}
        except Exception:
            return None, None

    def univariate_pattern_analysis(self, X, shap_values, feature_names=None):
        """
        Analyze univariate patterns between features and their SHAP values.

        For categorical features, performs group comparisons using t-tests, ANOVA,
        or non-parametric alternatives. For continuous features, fits linear,
        quadratic, and sigmoid models and selects the best based on RMSE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        shap_values : array-like of shape (n_samples, n_features)
            SHAP values for each sample and feature.
        feature_names : list of str, optional
            Names of features. If None, generates default names.

        Returns
        -------
        dict
            Dictionary mapping feature names to pattern analysis results.
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        X_df = pd.DataFrame(X, columns=feature_names)
        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        pattern_analysis = {}
        feature_types = {}

        for feature in self.results["significant_features"]:
            x = X_df[feature].values
            y = shap_df[feature].values
            f_type = self._get_feature_type(x)
            feature_types[feature] = f_type

            if f_type in ["binary", "discrete"]:
                groups = pd.Series(y).groupby(x)
                group_tests = {}
                between_p = None

                for val, group_y in groups:
                    _, p_shap = stats.shapiro(group_y)
                    if p_shap > 0.05:
                        _, p_val = stats.ttest_1samp(group_y, 0)
                    else:
                        _, p_val = stats.wilcoxon(group_y)
                    group_tests[val] = {"p_zero": p_val}

                group_list = [group_y for _, group_y in groups]
                shapiro_ps = [stats.shapiro(g)[1] for g in group_list if len(g) > 2]
                tukey_results = None
                if all(p > 0.05 for p in shapiro_ps):
                    if len(group_list) == 2:
                        _, between_p = stats.ttest_ind(*group_list)
                    else:
                        _, between_p = stats.f_oneway(*group_list)
                        if between_p < self.p_univariate:
                            tukey_result = pairwise_tukeyhsd(y, x)
                            tukey_results = {
                                "pairwise_comparisons": [],
                                "summary": str(tukey_result),
                            }
                            for i in range(len(tukey_result.groupsunique)):
                                for j in range(i + 1, len(tukey_result.groupsunique)):
                                    group1 = tukey_result.groupsunique[i]
                                    group2 = tukey_result.groupsunique[j]
                                    idx = (
                                        i * len(tukey_result.groupsunique)
                                        + j
                                        - ((i + 1) * (i + 2)) // 2
                                        + i
                                    )
                                    if idx < len(tukey_result.pvalues):
                                        tukey_results["pairwise_comparisons"].append(
                                            {
                                                "group1": group1,
                                                "group2": group2,
                                                "meandiff": tukey_result.meandiffs[idx],
                                                "p_adj": tukey_result.pvalues[idx],
                                                "lower": tukey_result.confint[idx][0],
                                                "upper": tukey_result.confint[idx][1],
                                                "reject": tukey_result.reject[idx],
                                            }
                                        )
                else:
                    _, between_p = stats.kruskal(*group_list)
                    if between_p < self.p_univariate and len(group_list) > 2:
                        from scipy.stats import ranksums

                        unique_vals = sorted(groups.groups.keys())
                        pairwise_comparisons = []
                        for i in range(len(unique_vals)):
                            for j in range(i + 1, len(unique_vals)):
                                group1_data = groups.get_group(unique_vals[i])
                                group2_data = groups.get_group(unique_vals[j])
                                _, p_val = ranksums(group1_data, group2_data)
                                pairwise_comparisons.append(
                                    {
                                        "group1": unique_vals[i],
                                        "group2": unique_vals[j],
                                        "p_value": p_val,
                                        "test": "wilcoxon_rank_sum",
                                    }
                                )
                        tukey_results = {
                            "pairwise_comparisons": pairwise_comparisons,
                            "summary": "Non-parametric pairwise comparisons using Wilcoxon rank-sum test",
                        }

                pattern_desc = (
                    f"{f_type.capitalize()} feature with significant differences between categories (p={between_p:.4f})"
                    if between_p < self.p_univariate
                    else "No significant pattern"
                )

                pattern_analysis[feature] = {
                    "pattern_description": pattern_desc,
                    "group_tests": group_tests,
                    "between_p": between_p,
                    "tukey_results": tukey_results,
                }

            elif f_type == "continuous":
                fits = {}
                rmse_lin, info_lin = self._fit_linear(x, y)
                if rmse_lin is not None:
                    fits["linear"] = (rmse_lin, info_lin)
                rmse_quad, info_quad = self._fit_quadratic(x, y)
                if rmse_quad is not None:
                    fits["quadratic"] = (rmse_quad, info_quad)
                rmse_sig, info_sig = self._fit_sigmoid(x, y)
                if rmse_sig is not None:
                    fits["sigmoid"] = (rmse_sig, info_sig)

                if fits:
                    best = min(fits, key=lambda k: fits[k][0])
                    info = fits[best][1]
                    self.logger.debug(
                        f"Feature '{feature}': Best fit is {best} with RMSE {fits[best][0]:.4f}"
                    )
                    if info["type"] == "linear":
                        dir_str = "Positive" if info["slope"] > 0 else "Negative"
                        pattern_desc = f"{dir_str} Linear: {'Higher' if info['slope'] > 0 else 'Lower'} feature values lead to {'higher' if info['slope'] > 0 else 'lower'} SHAP values"
                    elif info["type"] == "quadratic":
                        dir_str = "U-shaped" if info["a"] > 0 else "Inverted U-shaped"
                        pattern_desc = f"{dir_str}: Extreme values {'increase' if info['a'] > 0 else 'decrease'} SHAP values"
                    elif info["type"] == "sigmoid":
                        dir_str = "Increasing" if info["a"] > 0 else "Decreasing"
                        pattern_desc = f"{dir_str} Sigmoid: SHAP values saturate at {'high' if info['a'] > 0 else 'low'} feature values"
                else:
                    pattern_desc = "No significant pattern"

                pattern_analysis[feature] = {
                    "pattern_description": pattern_desc,
                    "best_fit": info if fits else None,
                    "all_fits": fits,
                    "tukey_results": None,  # Continuous features don't have Tukey tests
                }

        self.results["pattern_analysis"] = pattern_analysis
        self.results["feature_types"] = feature_types
        return pattern_analysis

    def interaction_analysis(self, X, shap_values, feature_names=None):
        """
        Detect and analyze feature interactions using SHAP interaction approximation.

        Uses SHAP's approximate_interactions to identify potential interactions,
        then validates through subgroup analysis and statistical testing.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        shap_values : array-like of shape (n_samples, n_features)
            SHAP values for each sample and feature.
        feature_names : list of str, optional
            Names of features. If None, generates default names.

        Returns
        -------
        dict
            Dictionary mapping interaction keys to interaction analysis results,
            sorted by interaction strength.
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        X_df = pd.DataFrame(X, columns=feature_names)
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        significant_features = self.results["significant_features"]
        feature_types = self.results["feature_types"]
        pattern_analysis = self.results["pattern_analysis"]

        interaction_results = {}

        self.logger.debug(
            f"Analyzing interactions for {len(significant_features)} significant features"
        )

        for target in significant_features:
            target_idx = list(feature_names).index(target)
            inter_indices = shap.utils.approximate_interactions(target_idx, shap_values, X)
            top_inters = [
                feature_names[j] for j in inter_indices[1 : self.max_interactions_per_feature + 1]
            ]

            for inter in top_inters:
                if inter not in significant_features:
                    continue
                key = f"{target}_x_{inter}"
                t_type = feature_types[target]
                i_type = feature_types[inter]

                x_t = X_df[target].values
                y_t = shap_df[target].values
                x_i = X_df[inter].values

                if t_type == "continuous" and i_type == "continuous":
                    mean_i = np.mean(x_i)
                    low_mask = x_i <= mean_i
                    high_mask = x_i > mean_i

                    best_fit_type = (
                        pattern_analysis[target]["best_fit"]["type"]
                        if pattern_analysis[target]["best_fit"]
                        else None
                    )
                    if not best_fit_type:
                        continue

                    def fit_group(mask, fit_type, x_vals, y_vals):
                        x_g, y_g = x_vals[mask], y_vals[mask]
                        if len(x_g) < 5:
                            return None, None
                        if fit_type == "linear":
                            return self._fit_linear(x_g, y_g)
                        elif fit_type == "quadratic":
                            return self._fit_quadratic(x_g, y_g)
                        elif fit_type == "sigmoid":
                            return self._fit_sigmoid(x_g, y_g)

                    rmse_low, info_low = fit_group(low_mask, best_fit_type, x_t, y_t)
                    rmse_high, info_high = fit_group(high_mask, best_fit_type, x_t, y_t)

                    if info_low is None and info_high is None:
                        continue

                    if info_low and info_high:
                        if best_fit_type == "linear":
                            y_pred_low = info_low["slope"] * x_t + info_low["intercept"]
                            y_pred_high = info_high["slope"] * x_t + info_high["intercept"]
                        elif best_fit_type == "quadratic":
                            y_pred_low = (
                                info_low["a"] * x_t**2 + info_low["b"] * x_t + info_low["c"]
                            )
                            y_pred_high = (
                                info_high["a"] * x_t**2 + info_high["b"] * x_t + info_high["c"]
                            )
                        elif best_fit_type == "sigmoid":
                            y_pred_low = (
                                info_low["L"]
                                / (1 + np.exp(-info_low["a"] * (x_t - info_low["x0"])))
                                + info_low["b"]
                            )
                            y_pred_high = (
                                info_high["L"]
                                / (1 + np.exp(-info_high["a"] * (x_t - info_high["x0"])))
                                + info_high["b"]
                            )

                        diffs_pred = y_pred_high - y_pred_low
                        _, p_shap_diff = stats.shapiro(diffs_pred)
                        if p_shap_diff > 0.05:
                            _, p_comp = stats.ttest_rel(y_pred_high, y_pred_low)
                        else:
                            _, p_comp = stats.wilcoxon(diffs_pred)

                        if abs(p_comp) < self.p_interaction:
                            mean_diff = np.mean(diffs_pred)
                            inter_type = "Synergistic" if mean_diff > 0 else "Antagonistic"

                            strength_raw = abs(mean_diff)

                            if self.normalize_interactions:
                                all_predictions = np.concatenate([y_pred_low, y_pred_high])
                                pred_range = np.max(all_predictions) - np.min(all_predictions)

                                if pred_range > 0:
                                    strength = (strength_raw / pred_range) * 100
                                else:
                                    shap_magnitude = np.mean(np.abs(y_t))
                                    strength = (
                                        (strength_raw / shap_magnitude) * 100
                                        if shap_magnitude > 0
                                        else 1.0
                                    )
                                strength = max(strength, 0.01)
                            else:
                                strength = strength_raw

                            desc = f"{inter_type} interaction: High {inter} {'amplifies' if mean_diff > 0 else 'attenuates'} the effect of {target}"
                            interaction_results[key] = {
                                "p_value": p_comp,
                                "interaction_strength": strength,
                                "interaction_type": inter_type,
                                "description": desc,
                            }

        sorted_interactions = dict(
            sorted(
                interaction_results.items(),
                key=lambda x: x[1]["interaction_strength"],
                reverse=True,
            )
        )

        self.results["interaction_analysis"] = sorted_interactions
        return sorted_interactions

    def comprehensive_analysis(self, X, shap_values, feature_names=None):
        """
        Run complete CLE-SH analysis pipeline.

        Executes the full analysis sequence: significance testing, univariate
        pattern analysis, and interaction detection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        shap_values : array-like of shape (n_samples, n_features)
            SHAP values for each sample and feature.
        feature_names : list of str, optional
            Names of features. If None, generates default names.

        Returns
        -------
        dict
            Complete analysis results dictionary.
        """
        self.logger.info("Starting Comprehensive SHAP Analysis (CLE-SH)")
        self.logger.info("=" * 60)

        self.logger.info("1. Statistical Significance Analysis")
        significant_features = self.analyze_shap_significance(shap_values, X, feature_names)
        self.logger.success(f"Found {len(significant_features)} statistically significant features")

        self.logger.info("2. Univariate Pattern Analysis")
        pattern_results = self.univariate_pattern_analysis(X, shap_values, feature_names)
        self.logger.success(f"Analyzed patterns for {len(pattern_results)} features")

        self.logger.info("3. Feature Interaction Analysis")
        interaction_results = self.interaction_analysis(X, shap_values, feature_names)
        self.logger.success(f"Found {len(interaction_results)} significant interactions")

        return self.results

    def generate_literal_explanations(self):
        """
        Generate human-readable explanations from analysis results.

        Creates natural language descriptions of significant features,
        their patterns, and interactions.

        Returns
        -------
        list of str
            List of explanation sentences.
        """
        explanations = []

        if not self.results or "significant_features" not in self.results:
            explanations.append(
                "No analysis has been performed yet. Please run comprehensive_analysis() first."
            )
            return explanations

        explanations.append(
            f"Statistical analysis identified {len(self.results['significant_features'])} features with significant impact on model predictions."
        )

        sig_analysis = self.results["significance_analysis"]
        sorted_features = sorted(
            self.results["significant_features"],
            key=lambda f: sig_analysis[f]["mean_abs_shap"],
            reverse=True,
        )
        if sorted_features:
            top_f = sorted_features[0]
            explanations.append(
                f"The most influential feature is '{top_f}' with average absolute SHAP value of {sig_analysis[top_f]['mean_abs_shap']:.4f}."
            )

        for f, data in self.results["pattern_analysis"].items():
            explanations.append(f"Feature '{f}': {data['pattern_description']}")

        for _, data in self.results["interaction_analysis"].items():
            explanations.append(
                f"Detected {data['interaction_type'].lower()} interaction: {data['description']}"
            )

        return explanations

    def visualize_results(self, figsize=(20, 16), custom_colors=None):
        """
        Generate professional visualization of CLE-SH analysis results.

        Creates a 2x2 subplot grid showing feature significance, type distribution,
        pattern types, and interaction strengths with professional styling.

        Parameters
        ----------
        figsize : tuple of float, default=(20, 16)
            Figure size as (width, height) in inches.
        custom_colors : dict, optional
            Custom color palette to override defaults. Keys should match
            the predefined color names (primary, secondary, etc.).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        colors = {
            "primary": "#2E4A6B",
            "secondary": "#5B7FA6",
            "accent": "#8FA4C7",
            "highlight": "#D4AF37",
            "success": "#2E7D5C",
            "warning": "#B8860B",
            "danger": "#A0522D",
            "neutral": "#6C7B7F",
            "light_gray": "#E8EAED",
            "dark_gray": "#4A5568",
        }

        if custom_colors:
            colors.update(custom_colors)

        qualitative_set = [
            colors["primary"],
            colors["highlight"],
            colors["success"],
            colors["warning"],
            colors["danger"],
            colors["secondary"],
        ]

        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 16,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": False,
            }
        )

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "CLE-SH: Comprehensive SHAP Analysis Results",
            fontsize=16,
            fontweight="bold",
            color=colors["dark_gray"],
            y=0.98,
        )

        sig_data = self.results["significance_analysis"]
        sig_features = self.results["significant_features"]
        mean_abs = [sig_data[f]["mean_abs_shap"] for f in sig_features]

        display_names = [f[:15] + "..." if len(f) > 15 else f for f in sig_features[:10]]

        bars = axes[0, 0].barh(
            display_names,
            mean_abs[:10],
            color=colors["primary"],
            edgecolor=colors["dark_gray"],
            linewidth=0.5,
        )
        axes[0, 0].set_xlabel("Mean Absolute SHAP Value", color=colors["dark_gray"])
        axes[0, 0].set_title(
            "Top 10 Significant Features", color=colors["dark_gray"], fontweight="bold"
        )
        axes[0, 0].tick_params(colors=colors["dark_gray"])
        axes[0, 0].grid(False)

        for bar in bars:
            width = bar.get_width()
            axes[0, 0].text(
                width * 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )

        if hasattr(self, "results") and "feature_types" in self.results:
            feature_types = self.results["feature_types"]
            type_counts = {}
            for feature in sig_features:
                ftype = feature_types.get(feature, "unknown")
                type_counts[ftype] = type_counts.get(ftype, 0) + 1

            if type_counts:
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                type_colors = qualitative_set[: len(types)]

                bars = axes[0, 1].bar(
                    types, counts, color=type_colors, edgecolor=colors["dark_gray"], linewidth=0.5
                )
                axes[0, 1].set_ylabel("Number of Features", color=colors["dark_gray"])
                axes[0, 1].set_title(
                    "Feature Type Distribution", color=colors["dark_gray"], fontweight="bold"
                )
                axes[0, 1].tick_params(colors=colors["dark_gray"])

                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        color=colors["dark_gray"],
                        fontweight="bold",
                    )

                axes[0, 1].set_xticklabels([t.capitalize() for t in types])
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No feature type data\navailable",
                    ha="center",
                    va="center",
                    color=colors["neutral"],
                    fontsize=11,
                    style="italic",
                )
                axes[0, 1].set_title(
                    "Feature Type Distribution", color=colors["dark_gray"], fontweight="bold"
                )
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No feature type data\navailable",
                ha="center",
                va="center",
                color=colors["neutral"],
                fontsize=11,
                style="italic",
            )
            axes[0, 1].set_title(
                "Feature Type Distribution", color=colors["dark_gray"], fontweight="bold"
            )

        if self.results["pattern_analysis"]:
            pattern_types = {}
            for _, data in self.results["pattern_analysis"].items():
                if data.get("best_fit") and "type" in data["best_fit"]:
                    ptype = data["best_fit"]["type"]
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                else:
                    pattern_types["no_pattern"] = pattern_types.get("no_pattern", 0) + 1

            if pattern_types:
                patterns = list(pattern_types.keys())
                pattern_counts = list(pattern_types.values())

                pattern_color_map = {
                    "linear": colors["primary"],
                    "quadratic": colors["highlight"],
                    "sigmoid": colors["success"],
                    "no_pattern": colors["neutral"],
                }
                pattern_colors = [
                    pattern_color_map.get(p, qualitative_set[i % len(qualitative_set)])
                    for i, p in enumerate(patterns)
                ]

                bars = axes[1, 0].bar(
                    patterns,
                    pattern_counts,
                    color=pattern_colors,
                    edgecolor=colors["dark_gray"],
                    linewidth=0.5,
                )
                axes[1, 0].set_ylabel("Number of Features", color=colors["dark_gray"])
                axes[1, 0].set_title(
                    "Univariate Pattern Types", color=colors["dark_gray"], fontweight="bold"
                )
                axes[1, 0].tick_params(axis="x", rotation=45, colors=colors["dark_gray"])
                axes[1, 0].tick_params(axis="y", colors=colors["dark_gray"])
                axes[1, 0].grid(False)

                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        color=colors["dark_gray"],
                        fontweight="bold",
                    )
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No pattern analysis\ndata available",
                    ha="center",
                    va="center",
                    color=colors["neutral"],
                    fontsize=11,
                    style="italic",
                )
                axes[1, 0].set_title(
                    "Univariate Pattern Types", color=colors["dark_gray"], fontweight="bold"
                )
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No pattern analysis\ndata available",
                ha="center",
                va="center",
                color=colors["neutral"],
                fontsize=11,
                style="italic",
            )
            axes[1, 0].set_title(
                "Univariate Pattern Types", color=colors["dark_gray"], fontweight="bold"
            )

        if self.results["interaction_analysis"]:
            inter_names = list(self.results["interaction_analysis"].keys())[:8]
            strengths = [
                self.results["interaction_analysis"][n]["interaction_strength"] for n in inter_names
            ]

            display_inter = [name.replace("_x_", " × ") for name in inter_names]

            def smart_truncate(name, max_len=35):
                if len(name) <= max_len:
                    return name
                parts = name.split(" × ")
                if len(parts) == 2:
                    part1, part2 = parts
                    max_part_len = (max_len - 3) // 2
                    if len(part1) > max_part_len:
                        part1 = part1[: max_part_len - 2] + ".."
                    if len(part2) > max_part_len:
                        part2 = part2[: max_part_len - 2] + ".."
                    return f"{part1} × {part2}"
                else:
                    return name[: max_len - 3] + "..."

            display_inter = [smart_truncate(name) for name in display_inter]

            colors_inter = []
            for name in inter_names:
                itype = self.results["interaction_analysis"][name]["interaction_type"]
                colors_inter.append(
                    colors["success"] if itype == "Synergistic" else colors["danger"]
                )

            bars = axes[1, 1].barh(
                display_inter,
                strengths,
                color=colors_inter,
                edgecolor=colors["dark_gray"],
                linewidth=0.5,
            )
            axes[1, 1].set_xlabel("Interaction Strength", color=colors["dark_gray"])
            axes[1, 1].set_title(
                "Top Feature Interactions", color=colors["dark_gray"], fontweight="bold"
            )
            axes[1, 1].tick_params(colors=colors["dark_gray"])

            for bar in bars:
                width = bar.get_width()
                if width >= 1:
                    label = f"{width:.1f}"
                elif width >= 0.1:
                    label = f"{width:.2f}"
                else:
                    label = f"{width:.3f}"

                axes[1, 1].text(
                    width * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=8,
                )

            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=colors["success"],
                    edgecolor=colors["dark_gray"],
                    linewidth=0.5,
                    label="Synergistic",
                ),
                Patch(
                    facecolor=colors["danger"],
                    edgecolor=colors["dark_gray"],
                    linewidth=0.5,
                    label="Antagonistic",
                ),
            ]
            legend = axes[1, 1].legend(
                handles=legend_elements,
                loc="lower right",
                fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor(colors["neutral"])
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No significant\ninteractions found",
                ha="center",
                va="center",
                color=colors["neutral"],
                fontsize=11,
                style="italic",
            )
            axes[1, 1].set_title(
                "Top Feature Interactions", color=colors["dark_gray"], fontweight="bold"
            )

        plt.tight_layout(rect=[0.12, 0.03, 1, 0.94])

        for ax in axes.flat:
            ax.set_facecolor("#FAFBFC")
            for spine in ax.spines.values():
                spine.set_color(colors["light_gray"])
                spine.set_linewidth(0.8)

        return fig
