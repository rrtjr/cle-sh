"""
CLE-SH Demo: Alternative with TreeExplainer

This example shows how to use CLE-SH with TreeExplainer in a more robust way,
with proper error handling and fallbacks.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import the CLESH class from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clesh.clesh import CLESH  # noqa: E402


def demo_clesh_with_tree_explainer():
    """
    Demonstrates CLE-SH analysis with TreeExplainer, with fallback to LinearExplainer.

    Returns:
        tuple: (clesh_analyzer, analysis_results) - The CLESH instance and results
    """
    print("CLE-SH Demo: Breast Cancer Dataset Analysis (with TreeExplainer)")
    print("=" * 60)

    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try TreeExplainer with RandomForest first
    print("Attempting to use TreeExplainer with RandomForest...")
    try:
        # Train RandomForest model
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
        rf_model.fit(X_train_scaled, y_train)
        print(f"RandomForest accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")

        # Try TreeExplainer with smaller dataset and error handling
        print("Generating SHAP values with TreeExplainer...")
        explainer = shap.TreeExplainer(rf_model)

        # Use a smaller subset to avoid memory/stability issues
        sample_size = min(50, len(X_test_scaled))
        X_sample = X_test_scaled[:sample_size]

        # Try with additivity check disabled
        shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_vals_class1 = shap_values[1]  # Binary classification, class 1
        elif len(shap_values.shape) == 3:
            shap_vals_class1 = shap_values[:, :, 1]
        else:
            shap_vals_class1 = shap_values

        print(f"SUCCESS: TreeExplainer succeeded! SHAP values shape: {shap_vals_class1.shape}")
        X_analysis = X_sample

    except Exception as e:
        print(f"FAILED: TreeExplainer failed: {e}")
        print("Falling back to LinearExplainer with LogisticRegression...")

        # Fallback to LinearExplainer
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        print(f"LogisticRegression accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")

        explainer = shap.LinearExplainer(lr_model, X_train_scaled)
        shap_vals_class1 = explainer.shap_values(X_test_scaled)

        print(f"SUCCESS: LinearExplainer succeeded! SHAP values shape: {shap_vals_class1.shape}")
        X_analysis = X_test_scaled

    # Run CLE-SH analysis with logging
    clesh = CLESH(alpha=0.01, interaction_threshold=0.1, log_level="INFO")
    results = clesh.comprehensive_analysis(X_analysis, shap_vals_class1, feature_names)

    # Generate and display literal explanations
    print("\nLiteral Explanations:")
    print("-" * 30)
    explanations = clesh.generate_literal_explanations()
    for i, explanation in enumerate(explanations, 1):
        print(f"{i}. {explanation}")

    # Generate visualization
    print("\nGenerating visualization...")
    fig = clesh.visualize_results()

    # Save plot instead of showing to avoid GUI issues
    try:
        fig.savefig("clesh_tree_explainer_results.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'clesh_tree_explainer_results.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")
    finally:
        plt.close(fig)  # Always close the figure

    return clesh, results


def print_detailed_results(analysis_results):
    """
    Print detailed analysis results in a formatted way.

    Args:
        analysis_results (dict): Results from CLE-SH analysis
    """
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 60)

    # Significant features
    print(f"\nSignificant Features ({len(analysis_results['significant_features'])}):")
    for feature in analysis_results["significant_features"][:10]:
        data = analysis_results["significance_analysis"][feature]
        print(f" • {feature}: mean |SHAP|={data['mean_abs_shap']:.4f}")

    # Top feature interactions
    print("\nTop Feature Interactions:")
    for interaction, data in list(analysis_results["interaction_analysis"].items())[:5]:
        strength = data["interaction_strength"]
        # Format strength based on magnitude for better readability
        if strength >= 1:
            strength_str = f"{strength:.1f}"
        elif strength >= 0.01:
            strength_str = f"{strength:.3f}"
        elif strength >= 0.001:
            strength_str = f"{strength:.4f}"
        else:
            strength_str = f"{strength:.2e}"  # Scientific notation for very small values
        print(
            f" • {interaction}: {data['interaction_type']} (strength={strength_str}, p={data['p_value']:.2e})"
        )


if __name__ == "__main__":
    # Run the demo
    clesh_analyzer, analysis_results = demo_clesh_with_tree_explainer()

    # Print detailed results
    print_detailed_results(analysis_results)
