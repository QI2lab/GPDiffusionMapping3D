"""Brownian predictions and ground-truth field recovery using actual CPU."""

import pytest
from scripts.validate_2d import brownian_checks, inference, SEEDS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("seed", SEEDS)
def test_constant_diffusion_components_msd_and_mle(seed):
    brownian_checks(seed)


@pytest.mark.parametrize(
    "seed,feature,substeps",
    [(seed, feature, 1) for seed in SEEDS for feature in (False, True)]
    + [(41, True, 10)],
)
def test_physical_map_recovery(seed, feature, substeps):
    assert inference(
        10000, "cpu", seed, feature, substeps=substeps
    )["recovery_rms"] < 0.1
