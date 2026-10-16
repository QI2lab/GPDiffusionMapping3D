"""Mathematical identities and CPU/CUDA comparisons on real hardware."""

import numpy as np
import pytest
from scipy import stats
import functions as f
from scripts.validate_2d import mathematical_checks

pytestmark=[pytest.mark.gpu,pytest.mark.unit]


def test_covariance_interpolation_prior_and_paired_acceptance(cupy_gpu):
    mathematical_checks()


def test_full_density_and_nonpositive_diffusion(cupy_gpu):
    cp=cupy_gpu
    diffusion=cp.array([.3,.6,.9])
    steps=np.array([[.1,-.2],[.5,.4],[-.3,.2]])
    a=cp.asarray(np.sum(steps**2,axis=1)/.12)
    expected=sum(stats.multivariate_normal.logpdf(x,cov=2*d*.03*np.eye(2)) for x,d in zip(steps,[.3,.6,.9]))
    np.testing.assert_allclose(float(f.log_likelihood(diffusion,a,.03,xp=cp)),expected,rtol=2e-14)
    assert float(f.gpu_likelihood_difference(cp.ones(3),cp.array([-2.,0.,0.]),1.,a))==-np.inf
