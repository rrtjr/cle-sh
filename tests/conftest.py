"""
Pytest configuration file for CLESH tests.

This file contains test configuration to ensure tests run reliably
by setting up appropriate backends and warnings.
"""

import os
import warnings

import pytest

# Set environment variables before any imports to prevent GUI issues
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""  # Disable display for headless testing

# Set matplotlib backend before any imports that might use it
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

# Filter out warnings that might clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):
    """Configure pytest with markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "visualization: marks tests that use matplotlib")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for each test."""
    # Ensure matplotlib uses non-interactive backend
    import matplotlib

    matplotlib.use("Agg", force=True)

    # Clear any existing matplotlib figures
    import matplotlib.pyplot as plt

    plt.close("all")

    yield

    # Clean up after test
    plt.close("all")
