"""Independent mathematical expectations for the real CPU implementation."""

import contextlib
import io
import numpy as np
import pytest
from scipy import stats

import functions as f
from scripts.validate_2d import simulate, parameters

pytestmark = [pytest.mark.cpu, pytest.mark.unit]


def test_gaussian_density_and_likelihood_difference():
    dt=.03
    steps=np.array([[.1,-.2],[.5,.4],[-.3,.2]])
    old=np.array([.3,.6,.9])
    new=np.array([.4,.5,1.1])
    a=np.sum(steps**2,axis=1)/(4*dt)
    oracle=lambda d: sum(stats.multivariate_normal.logpdf(x,mean=[0,0],cov=2*d0*dt*np.eye(2)) for x,d0 in zip(steps,d))
    np.testing.assert_allclose(f.log_likelihood(old,a,dt),oracle(old),rtol=2e-14)
    np.testing.assert_allclose(f.likelihood_difference(old,new,a),oracle(new)-oracle(old),rtol=2e-14,atol=2e-14)
    for invalid in (0.,-1.):
        assert f.log_likelihood(np.array([invalid]),a[:1],dt)==-np.inf
        assert f.likelihood_difference(old,np.array([invalid,1.,1.]),a)==-np.inf


def test_kernel_regularization_interpolation_and_prior():
    data=simulate(100)
    with contextlib.redirect_stdout(io.StringIO()):
        v=f.initialization(parameters(),data,backend='cpu')
    coords=v.induCoordinates
    kernel=np.array([[.25*np.exp(-sum((float(x)-float(y))**2 for x,y in zip(p,q))/(2*1.2**2)) for q in coords] for p in coords])
    np.testing.assert_allclose(v.cInduIndu,kernel,rtol=2e-14)
    np.testing.assert_array_equal(kernel,kernel.T)
    regularized=kernel+.01*kernel.mean()*np.eye(len(kernel))
    assert np.linalg.eigvalsh(regularized).min()>0
    tolerance=128*np.finfo(float).eps*np.linalg.cond(regularized)
    weights=np.linalg.solve(regularized,v.cInduData).T
    np.testing.assert_allclose(v.dData,weights@v.dIndu,rtol=tolerance,atol=tolerance)
    k,change=7,.02
    new=v.dIndu+v.cProposal[:,k]*change
    field=v.dData+v.cInduData[k]*change
    np.testing.assert_allclose(field,weights@new,rtol=tolerance,atol=tolerance)
    oracle=lambda d: stats.multivariate_normal.logpdf(d,mean=v.priorMean,cov=regularized)
    delta=f.log_prior(new,v.priorMean,v.cInduInduInv)-f.log_prior(v.dIndu,v.priorMean,v.cInduInduInv)
    np.testing.assert_allclose(delta,oracle(new)-oracle(v.dIndu),atol=tolerance)


@pytest.mark.parametrize('seed',[11,12,13])
def test_single_metropolis_proposal_against_full_density(seed):
    # One transformed coefficient: compute every random input and the target
    # density independently, including the conditional proposal correction.
    rng=np.random.RandomState(seed)
    rng.random_sample()  # Original indexShuffler consumes one uniform for m=1.
    old=np.array([1.2]); mean=np.array([1.]); covariance=np.array([[2.]])
    inverse=np.array([[.5]]); cross=np.array([[.4,.6]])
    field=np.array([.24,.36]); starts=np.zeros((2,2)); ends=np.array([[.2,.3],[.4,.5]])
    epsilon=np.array([.2]); dt=.5; temperature=2.
    alpha=.6
    scale=rng.exponential(.2); change=scale*alpha*rng.randn()
    new=old+covariance[:,0]*change; proposed=field+cross[0]*change
    density=lambda d,b: (sum(stats.multivariate_normal.logpdf(x,cov=2*b0*dt*np.eye(2)) for x,b0 in zip(ends,b))+stats.norm.logpdf(d[0],loc=1.,scale=np.sqrt(2)))
    old_p=density(old,field)
    prior_normalization=stats.norm.logpdf(1.,loc=1.,scale=np.sqrt(2))
    actual_p=old_p-prior_normalization
    expected_accept=False
    if np.all(new>0) and np.all(proposed>0):
        ratio=(density(new,proposed)-old_p)/temperature
        ratio+=stats.norm.logpdf(change,scale=abs(scale*(alpha+change)))-stats.norm.logpdf(change,scale=abs(scale*alpha))
        expected_accept=ratio>np.log(rng.rand())
    f.seed_sampler(seed)
    result=f.diffusionPointSamplerSA_nb(1,covariance,cross,inverse,cross.T@inverse,np.full(2,dt),starts,ends,starts,np.linalg.cholesky(covariance),old,actual_p,field,mean,.5,epsilon,temperature)
    np.testing.assert_allclose(result[0],new if expected_accept else old,rtol=2e-14)
    np.testing.assert_allclose(result[2],proposed if expected_accept else field,rtol=2e-14)
    assert bool(result[3][0])==expected_accept
