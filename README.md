# CLE-SH: Comprehensive Literal Explanation Package for SHapley Values

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2409.12578-b31b1b.svg)](https://arxiv.org/abs/2409.12578)

A Python library for statistically rigorous SHAP value interpretation. CLE-SH identifies significant features, discovers univariate patterns, and detects feature interactions using proper statistical testing.

## Installation

```bash
# Using uv (recommended)
uv add "cle-sh[all]"

# Or using pip
pip install "cle-sh[all]"
```

## Quick Start

```python
from clesh.clesh import CLESH
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data and train model
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Generate SHAP values
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Run CLE-SH analysis
clesh = CLESH()
results = clesh.comprehensive_analysis(X_test_scaled, shap_values, data.feature_names)

# Get explanations
explanations = clesh.generate_literal_explanations()
for exp in explanations:
    print(exp)

# Create visualization
fig = clesh.visualize_results()
fig.savefig('results.png')
```

## Key Features

- **Statistical Significance Testing**: Identifies genuinely important features using paired tests
- **Pattern Discovery**: Detects linear, quadratic, and sigmoid relationships automatically
- **Interaction Analysis**: Finds synergistic and antagonistic feature interactions
- **Human-Readable Output**: Generates natural language explanations
- **Professional Visualizations**: Creates publication-ready charts

## Configuration

```python
# Configure analysis parameters
clesh = CLESH(
    alpha=0.05,                      # Significance threshold
    p_univariate=0.05,              # Pattern significance
    p_interaction=0.05,             # Interaction significance
    candidate_num_min=10,           # Min features to consider
    candidate_num_max=20,           # Max features to consider
    log_level="INFO"                # Logging verbosity
)

# Custom visualization colors
fig = clesh.visualize_results(custom_colors={
    'primary': '#1B365D',
    'highlight': '#E6A800'
})
```

## Requirements

- Python 3.9+
- Core: `numpy`, `pandas`, `scipy`, `statsmodels`, `shap`, `loguru`
- Visualization: `matplotlib`, `seaborn` (included with `[all]`)
- Examples: `scikit-learn` (included with `[all]`)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Reference

Based on methodology from [arXiv:2409.12578](https://arxiv.org/abs/2409.12578)
