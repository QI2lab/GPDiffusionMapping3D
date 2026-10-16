"""Brownian predictions and ground-truth field recovery using actual CPU."""

import pytest
from scripts.validate_2d import brownian_checks, inference, SEEDS

pytestmark=[pytest.mark.cpu,pytest.mark.integration]


@pytest.mark.parametrize('seed',SEEDS)
def test_constant_diffusion_components_msd_and_mle(seed):
    brownian_checks(seed)


@pytest.mark.parametrize('seed',SEEDS)
@pytest.mark.parametrize('feature',[False,True])
def test_physical_map_recovery(seed,feature):
    result=inference(10000,'cpu',seed,feature)
    assert result['recovery_rms'] < .1


def test_finer_simulation_preserves_smooth_recovery():
    assert inference(10000,'cpu',41,True,substeps=10)['recovery_rms'] < .1
