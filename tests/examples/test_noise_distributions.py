import numpy as np
import pandas as pd
import pytest
from estimagic.examples.benchmarking import _sample_from_distribution
from estimagic.examples.noise_distributions import NOISE_DISTRIBUTIONS


@pytest.mark.parametrize("distribution", NOISE_DISTRIBUTIONS)
def test_sample_from_distribution(distribution):
    np.random.seed(1234)
    mean = 0.33
    std = 0.55
    correlation = 0.44
    sample = _sample_from_distribution(
        distribution=distribution,
        mean=mean,
        std=std,
        size=(100_000, 5),
        correlation=correlation,
    )
    calculated_mean = sample.mean()
    calculated_std = sample.std()
    corrmat = pd.DataFrame(sample).corr().to_numpy().round(2)
    calculated_avgcorr = corrmat[~np.eye(len(corrmat)).astype(bool)].mean()

    assert np.allclose(calculated_mean, mean, atol=0.001)
    assert np.allclose(calculated_std, std, atol=0.001)
    assert np.allclose(calculated_avgcorr, correlation, atol=0.001)
