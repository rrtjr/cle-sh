"""
Test suite for the CLESH (Comprehensive Literal Explanation for SHapley values) package.

This module contains comprehensive unit tests and integration tests for all
functionality in the CLESH package.
"""

import os
import sys

# Set matplotlib backend for testing to avoid GUI issues
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to import clesh
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clesh.clesh import CLESH


class TestCLESHInitialization:
    """Test CLESH class initialization and parameters."""

    def test_default_initialization(self):
        """Test CLESH initialization with default parameters."""
        clesh = CLESH()
        assert clesh.alpha == 0.05
        assert clesh.p_univariate == 0.05
        assert clesh.p_interaction == 0.05
        assert clesh.candidate_num_min == 10
        assert clesh.candidate_num_max == 20
        assert clesh.interaction_threshold == 0.01
        assert clesh.max_interactions_per_feature == 3
        assert clesh.results == {}

    def test_custom_initialization(self):
        """Test CLESH initialization with custom parameters."""
        clesh = CLESH(
            alpha=0.01,
            p_univariate=0.02,
            p_interaction=0.03,
            candidate_num_min=5,
            candidate_num_max=15,
            interaction_threshold=0.05,
            max_interactions_per_feature=5,
        )
        assert clesh.alpha == 0.01
        assert clesh.p_univariate == 0.02
        assert clesh.p_interaction == 0.03
        assert clesh.candidate_num_min == 5
        assert clesh.candidate_num_max == 15
        assert clesh.interaction_threshold == 0.05
        assert clesh.max_interactions_per_feature == 5


class TestFeatureTypeDetection:
    """Test feature type detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clesh = CLESH()

    def test_binary_feature_detection(self):
        """Test detection of binary features."""
        binary_feature = np.array([0, 1, 0, 1, 1, 0, 1])
        assert self.clesh._get_feature_type(binary_feature) == "binary"

        # Test with two unique values
        binary_feature2 = np.array([5, 10, 5, 10, 5])
        assert self.clesh._get_feature_type(binary_feature2) == "binary"

    def test_discrete_feature_detection(self):
        """Test detection of discrete features."""
        discrete_feature = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        assert self.clesh._get_feature_type(discrete_feature) == "discrete"

        # Test with exactly 20 unique values
        discrete_feature2 = np.arange(20)
        assert self.clesh._get_feature_type(discrete_feature2) == "discrete"

    def test_continuous_feature_detection(self):
        """Test detection of continuous features."""
        continuous_feature = np.random.random(100)
        assert self.clesh._get_feature_type(continuous_feature) == "continuous"

        # Test with more than 20 unique values
        continuous_feature2 = np.arange(25)
        assert self.clesh._get_feature_type(continuous_feature2) == "continuous"


class TestStatisticalSignificanceAnalysis:
    """Test statistical significance analysis functionality."""

    def setup_method(self):
        """Set up test fixtures with synthetic data."""
        self.clesh = CLESH(alpha=0.05, candidate_num_min=3, candidate_num_max=8)

        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create SHAP values with varying importance
        # First few features have higher SHAP values
        self.shap_values = np.random.randn(self.n_samples, self.n_features)
        for i in range(3):
            self.shap_values[:, i] *= 3 - i  # Make first features more important

        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

    def test_significance_analysis_basic(self):
        """Test basic functionality of significance analysis."""
        significant_features = self.clesh.analyze_shap_significance(
            self.shap_values, self.X, self.feature_names
        )

        # Check that we get some significant features
        assert len(significant_features) >= self.clesh.candidate_num_min
        assert len(significant_features) <= self.clesh.candidate_num_max

        # Check that results are stored
        assert "significance_analysis" in self.clesh.results
        assert "significant_features" in self.clesh.results

        # Check that all features have significance analysis
        assert len(self.clesh.results["significance_analysis"]) == self.n_features

    def test_significance_analysis_feature_ranking(self):
        """Test that features are ranked by mean absolute SHAP values."""
        significant_features = self.clesh.analyze_shap_significance(
            self.shap_values, self.X, self.feature_names
        )

        # Get mean absolute SHAP values for significant features
        sig_analysis = self.clesh.results["significance_analysis"]
        shap_values = [sig_analysis[f]["mean_abs_shap"] for f in significant_features]

        # Check that they are in descending order
        assert shap_values == sorted(shap_values, reverse=True)

    def test_significance_analysis_with_array_feature_names(self):
        """Test significance analysis with numpy array feature names."""
        feature_names_array = np.array(self.feature_names)
        significant_features = self.clesh.analyze_shap_significance(
            self.shap_values, self.X, feature_names_array
        )

        assert len(significant_features) >= self.clesh.candidate_num_min
        assert all(isinstance(f, str) for f in significant_features)


class TestUnivariatePatternAnalysis:
    """Test univariate pattern analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clesh = CLESH()
        np.random.seed(42)

        # Create synthetic data with known patterns
        self.n_samples = 100
        self.X = np.random.randn(self.n_samples, 5)

        # Create SHAP values with known patterns
        self.shap_values = np.zeros((self.n_samples, 5))

        # Linear relationship
        self.shap_values[:, 0] = 2 * self.X[:, 0] + np.random.normal(0, 0.1, self.n_samples)

        # Quadratic relationship
        self.shap_values[:, 1] = self.X[:, 1] ** 2 + np.random.normal(0, 0.1, self.n_samples)

        # Binary feature
        self.X[:, 2] = np.random.choice([0, 1], self.n_samples)
        self.shap_values[:, 2] = np.where(self.X[:, 2] == 1, 1, -1) + np.random.normal(
            0, 0.1, self.n_samples
        )

        self.feature_names = [f"feature_{i}" for i in range(5)]

        # Run significance analysis first
        self.clesh.analyze_shap_significance(self.shap_values, self.X, self.feature_names)

    def test_linear_pattern_fitting(self):
        """Test linear pattern fitting."""
        x = np.linspace(-2, 2, 50)
        y = 2 * x + np.random.normal(0, 0.1, 50)

        rmse, info = self.clesh._fit_linear(x, y)

        assert rmse is not None
        assert info is not None
        assert info["type"] == "linear"
        assert abs(info["slope"] - 2) < 0.5  # Should be close to 2
        assert info["p_a"] < 0.05  # Should be significant

    def test_quadratic_pattern_fitting(self):
        """Test quadratic pattern fitting."""
        x = np.linspace(-2, 2, 50)
        y = x**2 + np.random.normal(0, 0.1, 50)

        rmse, info = self.clesh._fit_quadratic(x, y)

        assert rmse is not None
        assert info is not None
        assert info["type"] == "quadratic"
        assert abs(info["a"] - 1) < 0.5  # Should be close to 1
        assert info["p_a"] < 0.05  # Should be significant

    def test_sigmoid_pattern_fitting(self):
        """Test sigmoid pattern fitting."""
        x = np.linspace(-5, 5, 50)
        y = 1 / (1 + np.exp(-x)) + np.random.normal(0, 0.05, 50)

        rmse, info = self.clesh._fit_sigmoid(x, y)

        # Sigmoid fitting might fail sometimes due to optimization
        if rmse is not None:
            assert info["type"] == "sigmoid"
            assert "a" in info
            assert "p_a" in info

    def test_univariate_pattern_analysis(self):
        """Test complete univariate pattern analysis."""
        pattern_results = self.clesh.univariate_pattern_analysis(
            self.X, self.shap_values, self.feature_names
        )

        # Check that we have pattern analysis for significant features
        assert len(pattern_results) > 0
        assert "pattern_analysis" in self.clesh.results
        assert "feature_types" in self.clesh.results

        # Check that each result has required fields
        for _, result in pattern_results.items():
            assert "pattern_description" in result
            assert isinstance(result["pattern_description"], str)


class TestInteractionAnalysis:
    """Test interaction analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clesh = CLESH(max_interactions_per_feature=2)
        np.random.seed(42)

        # Create synthetic data
        self.n_samples = 100
        self.n_features = 6
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.shap_values = np.random.randn(self.n_samples, self.n_features)
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Run prerequisite analyses
        self.clesh.analyze_shap_significance(self.shap_values, self.X, self.feature_names)
        self.clesh.univariate_pattern_analysis(self.X, self.shap_values, self.feature_names)

    def test_interaction_analysis_basic(self):
        """Test basic interaction analysis functionality."""
        interaction_results = self.clesh.interaction_analysis(
            self.X, self.shap_values, self.feature_names
        )

        # Check that results are stored
        assert "interaction_analysis" in self.clesh.results

        # Check that interaction results have proper structure
        for key, result in interaction_results.items():
            assert "_x_" in key  # Interaction key format
            assert "interaction_type" in result
            assert "interaction_strength" in result
            assert "description" in result
            assert result["interaction_type"] in ["Synergistic", "Antagonistic"]


class TestComprehensiveAnalysis:
    """Test the complete comprehensive analysis workflow."""

    def setup_method(self):
        """Set up test fixtures with synthetic data to avoid SHAP issues."""
        # Use synthetic data instead of real SHAP to avoid segfaults
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 6

        # Create synthetic data that mimics real patterns
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create synthetic SHAP values with realistic patterns
        self.shap_values = np.random.randn(self.n_samples, self.n_features)
        # Make first few features more important
        for i in range(3):
            self.shap_values[:, i] *= 3 - i

        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        self.clesh = CLESH(alpha=0.1, candidate_num_min=3, candidate_num_max=5)

    def test_comprehensive_analysis_workflow(self):
        """Test the complete comprehensive analysis workflow."""
        results = self.clesh.comprehensive_analysis(self.X, self.shap_values, self.feature_names)

        # Check that all expected components are present
        assert "significant_features" in results
        assert "significance_analysis" in results
        assert "pattern_analysis" in results
        assert "interaction_analysis" in results

        # Check that we have some significant features
        assert len(results["significant_features"]) >= self.clesh.candidate_num_min

        # Check that pattern analysis was performed
        assert len(results["pattern_analysis"]) > 0

        # Check that literal explanations can be generated
        explanations = self.clesh.generate_literal_explanations()
        assert len(explanations) > 0
        assert all(isinstance(exp, str) for exp in explanations)

    @pytest.mark.visualization
    def test_visualization_generation(self):
        """Test that visualization can be generated without errors."""
        # Run analysis first
        self.clesh.comprehensive_analysis(self.X, self.shap_values, self.feature_names)

        # Generate visualization (should not raise errors)
        try:
            fig = self.clesh.visualize_results()
            assert fig is not None
            # Clean up
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Visualization generation failed: {e}")


@pytest.mark.integration
class TestRealDataIntegration:
    """Test with real data and real SHAP values."""

    def test_with_breast_cancer_and_linear_explainer(self):
        """Test with real breast cancer data and LinearExplainer (stable)."""
        # Load and prepare data
        data = load_breast_cancer()
        X, y = data.data[:100, :10], data.target[:100]  # Reasonable subset
        feature_names = data.feature_names[:10]

        from sklearn.linear_model import LogisticRegression

        # Train model and generate real SHAP values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use LogisticRegression + LinearExplainer (very stable)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)

        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

        # Test CLE-SH analysis with real SHAP values
        clesh = CLESH(alpha=0.2, candidate_num_min=2, candidate_num_max=6)
        results = clesh.comprehensive_analysis(X_test_scaled, shap_values, feature_names)

        # Validate results
        assert len(results["significant_features"]) >= 2
        assert len(results["pattern_analysis"]) > 0
        assert "interaction_analysis" in results

        # Test explanation generation
        explanations = clesh.generate_literal_explanations()
        assert len(explanations) > 0
        assert any("feature" in exp.lower() for exp in explanations)

    @pytest.mark.slow
    def test_with_permutation_explainer(self):
        """Test with PermutationExplainer (slower but model-agnostic)."""
        # Very small dataset for permutation explainer
        data = load_breast_cancer()
        X, y = data.data[:50, :5], data.target[:50]
        feature_names = data.feature_names[:5]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        model.fit(X_scaled, y)

        # Use PermutationExplainer (model-agnostic)
        explainer = shap.PermutationExplainer(model.predict_proba, X_scaled[:20])
        shap_values = explainer.shap_values(X_scaled[:10])  # Very small sample

        # Handle PermutationExplainer output (3D for binary classification)
        if len(shap_values.shape) == 3:
            shap_vals_class1 = shap_values[:, :, 1]
        else:
            shap_vals_class1 = shap_values

        # Test CLE-SH analysis
        clesh = CLESH(alpha=0.3, candidate_num_min=2, candidate_num_max=4)
        results = clesh.comprehensive_analysis(X_scaled[:10], shap_vals_class1, feature_names)

        # Basic validation
        assert len(results["significant_features"]) >= 2
        assert "pattern_analysis" in results


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clesh = CLESH()

    def test_empty_shap_values(self):
        """Test behavior with empty SHAP values."""
        with pytest.raises((ValueError, IndexError)):
            self.clesh.analyze_shap_significance(np.array([]), np.array([]), [])

    def test_single_feature(self):
        """Test behavior with single feature."""
        X = np.random.randn(50, 1)
        shap_values = np.random.randn(50, 1)
        feature_names = ["single_feature"]

        # Should handle single feature gracefully
        significant_features = self.clesh.analyze_shap_significance(shap_values, X, feature_names)
        assert len(significant_features) == 1

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test negative alpha
        clesh = CLESH(alpha=-0.1)
        assert clesh.alpha == -0.1  # Should accept but might cause issues in analysis

        # Test min > max
        clesh = CLESH(candidate_num_min=20, candidate_num_max=10)
        assert clesh.candidate_num_min == 20
        assert clesh.candidate_num_max == 10

    def test_generate_literal_explanations_empty_results(self):
        """Test literal explanation generation with empty results."""
        explanations = self.clesh.generate_literal_explanations()
        # Should return some default explanation even with no results
        assert isinstance(explanations, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_shap_values(self):
        """Test with constant SHAP values (no variation)."""
        clesh = CLESH(candidate_num_min=1, candidate_num_max=3)
        X = np.random.randn(50, 5)
        shap_values = np.ones((50, 5))  # All constant
        feature_names = [f"feature_{i}" for i in range(5)]

        # Should handle constant values gracefully
        significant_features = clesh.analyze_shap_significance(shap_values, X, feature_names)
        assert len(significant_features) >= 1

    def test_perfectly_correlated_features(self):
        """Test with perfectly correlated features."""
        clesh = CLESH()
        X = np.random.randn(50, 3)
        X[:, 1] = X[:, 0]  # Make feature 1 identical to feature 0

        shap_values = np.random.randn(50, 3)
        shap_values[:, 1] = shap_values[:, 0]  # Make SHAP values identical too

        feature_names = ["feature_0", "feature_1", "feature_2"]

        # Should handle correlated features
        significant_features = clesh.analyze_shap_significance(shap_values, X, feature_names)
        assert len(significant_features) >= 1

    def test_nan_values_handling(self):
        """Test handling of NaN values in data."""
        clesh = CLESH()
        X = np.random.randn(50, 3)
        shap_values = np.random.randn(50, 3)

        # Introduce some NaN values
        X[0, 0] = np.nan
        shap_values[1, 1] = np.nan

        feature_names = ["feature_0", "feature_1", "feature_2"]

        # Should either handle NaN gracefully or raise appropriate error
        try:
            significant_features = clesh.analyze_shap_significance(shap_values, X, feature_names)
            # If it succeeds, that's fine
            assert isinstance(significant_features, list)
        except Exception:
            # If it fails, that's also acceptable for NaN handling
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
