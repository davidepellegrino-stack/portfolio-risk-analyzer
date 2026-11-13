import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def normal_returns():
    """3,000 i.i.d. N(0, 1%) daily returns: a world where the normal model is correct."""
    rng = np.random.default_rng(11)
    return pd.Series(rng.normal(0, 0.01, 3000), index=pd.bdate_range("2012-01-02", periods=3000))
