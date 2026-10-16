"""Actual CUDA posterior sampling against independently integrated densities."""

import pytest
from scripts.posterior_oracle import validate

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("dimensions", [1, 2])
def test_small_positive_diffusion_posterior(cupy_gpu, dimensions):
    validate("gpu", dimensions)
