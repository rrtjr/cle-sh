"""
CLE-SH Demo: Breast Cancer Dataset Analysis

This example demonstrates how to use the CLE-SH package for comprehensive
SHAP analysis including statistical significance testing, univariate pattern
analysis, and feature interaction detection.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Import the CLESH class from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clesh.clesh import CLESH  # noqa: E402


def demo_clesh_analysis():
    """
    Demonstrates CLE-SH analysis on the breast cancer dataset.

    Returns:
        tuple: (clesh_analyzer, analysis_results) - The CLESH instance and results
    """
    print("CLE-SH Demo: Breast Cancer Dataset Analysis")
    print("=" * 50)

    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model - Using LogisticRegression for stable SHAP LinearExplainer
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print(f"Model accuracy: {model.score(X_test_scaled, y_test):.3f}")

    # Generate SHAP values using LinearExplainer (much more stable than TreeExplainer)
    print("\nGenerating SHAP values...")
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # LinearExplainer returns a 2D array directly for binary classification
    shap_vals_class1 = shap_values

    # Run CLE-SH analysis with logging
    clesh = CLESH(alpha=0.01, interaction_threshold=0.1, log_level="INFO")
    results = clesh.comprehensive_analysis(X_test_scaled, shap_vals_class1, feature_names)

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
        fig.savefig("clesh_analysis_results.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'clesh_analysis_results.png'")
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
    clesh_analyzer, analysis_results = demo_clesh_analysis()

    # Print detailed results
    print_detailed_results(analysis_results)
