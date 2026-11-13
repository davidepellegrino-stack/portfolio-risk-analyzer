import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def asset_returns():
    """Three correlated synthetic daily return series (fixed seed, no network)."""
    rng = np.random.default_rng(7)
    vols = np.array([0.010, 0.015, 0.022])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.4],
                     [0.2, 0.4, 1.0]])
    cov = np.outer(vols, vols) * corr
    mu = np.array([0.0004, 0.0005, 0.0006])
    data = rng.multivariate_normal(mu, cov, size=1500)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


@pytest.fixture(scope="session")
def portfolio_returns(asset_returns):
    return asset_returns.mean(axis=1)
