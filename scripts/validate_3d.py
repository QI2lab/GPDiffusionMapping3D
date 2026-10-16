"""Independent timestamped 3D physics and predeclared recovery protocol."""

import contextlib
import io
from pathlib import Path
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import data_reader
import functions as f
import model

SEEDS=(41,42,43)


def truth(points,feature=False):
    """Define an unbounded smooth positive 3D diffusion field in nm²/s."""
    return 1 + (.4*np.exp(-np.sum(points**2,axis=1)/2) if feature else np.zeros(len(points)))


def simulate(count=20000,seed=41,feature=False,irregular=True,substeps=1):
    """Simulate independent linked ZYX pairs with known actual elapsed time."""
    rng=np.random.default_rng(seed)
    start=rng.uniform(-3,3,(count,3))
    dt=rng.uniform(.002,.008,count) if irregular else np.full(count,.005)
    end=start.copy()
    for _ in range(substeps):
        end+=np.sqrt(2*truth(end,feature)[:,None]*dt[:,None]/substeps)*rng.normal(size=end.shape)
    positions=np.stack((start,end),axis=1).reshape(-1,3)
    ids=np.repeat(np.arange(count),2)
    times=np.stack((np.zeros(count),dt),axis=1).reshape(-1)
    return ids,positions,times


def prepared(count=20000,seed=41,feature=False,irregular=True,substeps=1):
    """Prepare physical transitions with the same path as user tracks."""
    return data_reader.prepare_tracks(*simulate(count,seed,feature,irregular,substeps))


def physics(seed=41,irregular=True):
    """Use independent chi-square and Gaussian expectations in three axes."""
    ids,pos,times=simulate(seed=seed,irregular=irregular)
    steps=pos[1::2]-pos[::2]
    tau=times[1::2]-times[::2]
    normalized=steps/np.sqrt(2*tau[:,None])
    n=len(steps)
    for component in normalized.T:
        assert abs(component.mean())<6/np.sqrt(n)
        assert stats.chi2.ppf(.0005,n)<sum(component**2)<stats.chi2.ppf(.9995,n)
    total=np.sum(normalized**2)
    assert stats.chi2.ppf(.0005,3*n)<total<stats.chi2.ppf(.9995,3*n)
    data=data_reader.prepare_tracks(ids,pos,times)
    a=np.sum((data.ends-data.starts)**2,axis=1)/(4*data.delta_t)
    expected=sum(stats.multivariate_normal.logpdf(step,cov=2*t*np.eye(3)) for step,t in zip(steps,tau))
    np.testing.assert_allclose(f.log_likelihood(np.ones(n),a,data.delta_t,3),expected,rtol=128*np.finfo(float).eps*n)
    mle=np.mean(np.sum(steps**2,axis=1)/(6*tau))
    assert abs(mle-1)<max(abs(stats.chi2.ppf(q,3*n)/(3*n)-1) for q in (.0005,.9995))
    return dict(msd_per_time=float(np.mean(np.sum(steps**2,axis=1)/tau)),expected=6.,mle=float(mle))


def recovery(backend='cpu',seed=41,feature=False,irregular=True,substeps=1):
    """Recover a predeclared central cube with the original sampler schedule."""
    data=prepared(seed=seed,feature=feature,irregular=irregular,substeps=substeps)
    if backend=='gpu':
        import cupy as cp
        cp.cuda.Device().synchronize()
    start=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        v,samples,p=model.analyze(data,nIter=100,backend=backend,seed=seed,nInduX=5,nInduY=5,nInduZ=5,
            nFineX=5,nFineY=5,nFineZ=5,covLambda=.5,covL=1.5)
    if backend=='gpu': cp.cuda.Device().synchronize()
    points=np.array([(z,y,x) for z in (-1.,0.,1.) for y in (-1.,0.,1.) for x in (-1.,0.,1.)])
    weights=f.covMat(points,v.induCoordinates,v.covLambda,v.covL)@v.cInduInduInv
    estimate=weights@v.dIndu
    error=float(np.sqrt(np.mean(((estimate-truth(points,feature))/truth(points,feature))**2)))
    return dict(backend=backend,seed=seed,feature=feature,irregular=irregular,substeps=substeps,
                seconds=time.perf_counter()-start,recovery_rms=error,estimate=estimate.tolist(),
                gpu_pool_bytes=cp.get_default_memory_pool().total_bytes() if backend=='gpu' else 0)
