# CLE-SH: Comprehensive Literal Explanation Package for SHapley Values by Statistical Validity

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2409.12578-b31b1b.svg)](https://arxiv.org/abs/2409.12578)

## Overview

CLE-SH is a Python library that boosts the interpretability of SHAP values with better statistical backing. It sticks closely to the approach in the [reference paper](https://arxiv.org/abs/2409.12578), keeping core ideas intact—like identifying "statistically significant features" through ranked mean absolute SHAP differences using paired tests adjusted for normality, defining "univariate patterns" via type-specific fits or group stats with p-value cutoffs, and spotting "feature interactions" with SHAP's approximate interactions plus conditional group validations.

The library sidesteps typical SHAP pitfalls, such as jumping to linear assumptions or ignoring non-normal distributions. It uses Shapiro-Wilk tests to check normality and picks the best fit (linear, quadratic, or sigmoid) based on RMSE for continuous features.

Tried testing it on biomedical tasks, like pulling SHAP insights from random forests on the breast cancer dataset, but it works for any field needing reliable XAI. It outputs clear, human-friendly explanations grounded in stats, which helps with reproducibility and cuts down on guesswork.

Key aspects:
- **Statistical Rigor**: Ranking-based tests cap significant features at 10-20 by default to avoid overload.
- **Pattern Handling**: Continuous features get functional fits (e.g., sigmoid via curve_fit with t-distributed p-value approximations) for accurate non-linearity; binary/discrete use ANOVA/Kruskal-Wallis and Tukey HSD for group differences.
- **Interaction Detection**: Targets top interactors per feature using `shap.utils.approximate_interactions`, then confirms with subgroup fits and paired tests to separate real effects (like synergistic boosts) from noise.

Edge cases are covered, including switching to Wilcoxon for non-normal SHAP values and skipping fits on tiny subgroups (<5 samples), aligning with the paper's focus on validity.

## Installation

Install CLE-SH via pip from PyPI (once published) or straight from GitHub.

### From PyPI (Recommended, when available)
```bash
pip install clesh
```

### From Source
```bash
git clone https://github.com/rrtjr/cle-sh.git
cd cle-sh
pip install -e .
```

### Requirements
- Python 3.8 or later
- Main dependencies: numpy, pandas, scipy, statsmodels, shap, matplotlib, seaborn
- For demos: scikit-learn

```bash
pip install -r requirements.txt
```

No extra installs needed beyond these—the library leans on standard stats tools to keep things light.

## Usage

### Basic Pipeline
Load the CLESH class, feed it your data and SHAP values, and run the full analysis.

```python
import numpy as np
from clesh import CLESH
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Train model and generate SHAP
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)[1]  # Class 1 for binary

# Run CLE-SH
clesh = CLESH(alpha=0.01, p_univariate=0.05, p_interaction=0.05,
              candidate_num_min=10, candidate_num_max=20,
              max_interactions_per_feature=3)
results = clesh.comprehensive_analysis(X_test_scaled, shap_values, feature_names)

# Generate explanations
explanations = clesh.generate_literal_explanations()
for exp in explanations:
    print(exp)

# Visualize
fig = clesh.visualize_results()
fig.savefig('clesh_results.png')
```

This produces explanations, like "Feature 'worst radius': Positive Linear: Higher feature values lead to higher SHAP values," where the pattern holds to a significant linear fit.

### Advanced Configuration
- **Hyperparameters**: Adjust `alpha` to tighten feature selection, `p_univariate` for stricter patterns, or `max_interactions_per_feature` to dial in interaction scope.
- **Extensions**: It auto-detects feature types for mixed data; tweak `_fit_*` methods if you want custom fits.
- **Validation Tip**: Check the results for normality flags—if Wilcoxon's kicking in often, think about transforming your data.

## Features
- **Significance Analysis**: Ranks by mean |SHAP|, picks via adjacent paired tests (t-test or Wilcoxon signed-rank), and limits to a sensible range of candidates.
- **Univariate Patterns**: Adapts to types—RMSE-selected fits (linear/quadratic/sigmoid with p-thresholds) for continuous; one-sample and ANOVA/Kruskal + Tukey for binary/discrete.
- **Interaction Detection**: Grabs top interactors from SHAP utils, validates with subgroup fits and pairs (e.g., calls it synergistic if the high-group slope outpaces the low).
- **Literal Explanations**: Straightforward summaries that keep the stats front and center.
- **Visualizations**: Bars for significance and interactions; easy to add dependence plots with fits.
- **Modularity**: Call individual methods like `analyze_shap_significance` as needed.

## Contributing
Happy to take contributions for better alignment or new bits (say, handling ordered discretes). Just:
1. Fork the repo.
2. Branch out: `git checkout -b feature/AmazingFeature`.
3. Commit: `git commit -m 'Add some AmazingFeature'`.
4. Push: `git push origin feature/AmazingFeature`.
5. Pull request.

Include tests that check definitions (e.g., fits minimize RMSE as in paper examples) and aim for full coverage.

## License
MIT License—see `LICENSE` for the full text.

## References
- [web] https://arxiv.org/abs/2409.12578
