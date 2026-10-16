"""Evaluate experimental all-float32 GPU arithmetic; never certify by memory alone."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import cupy as cp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import functions as f
import model
from validate_2d import simulate,parameters,truth,SEEDS

# Predeclared acceptance-error budget: |delta log acceptance| <= 1e-4.
# This bounds the multiplicative acceptance probability perturbation by exp(1e-4).
RATIO_BUDGET=1e-4
parser=argparse.ArgumentParser()
parser.add_argument("--dimensions",type=int,choices=(2,3),default=2)
parser.add_argument("--count",type=int,default=10000)
parser.add_argument("--skip-inference",action="store_true")
parser.add_argument("--output",default="docs/stage2-results.json")
args=parser.parse_args()
results=[]
for stress in (False,True):
    if args.dimensions == 3:
        from validate_3d import prepared
        data = prepared(count=args.count)
    else:
        data=simulate(args.count)
    p=parameters()
    if args.dimensions == 3:
        p.update(nInduX=5,nInduY=5,nInduZ=5,nFineX=5,nFineY=5,nFineZ=5,covL=1.5)
    dt=data.delta_t if args.dimensions==3 else data.deltaT
    if stress: p['covL']=100.
    with contextlib.redirect_stdout(io.StringIO()):
        cpu=f.initialization(p,data)
        reference=f.initialization(p,data,backend='gpu')
    for dtype in ('float64','float32'):
        with contextlib.redirect_stdout(io.StringIO()):
            start=time.perf_counter()
            v=f.initialization(p,data,backend='gpu',dtype=dtype)
            cp.cuda.Device().synchronize()
            elapsed=time.perf_counter()-start
        identity=cp.eye(v.nIndu,dtype=v.cProposal.dtype)
        residual=float(cp.linalg.norm(v.cProposal@v.cInduInduInv-identity,ord=cp.inf))
        live=sum(x.nbytes for x in vars(v).values() if isinstance(x,cp.ndarray))
        ll=float(f.log_likelihood(v.dData,v.a,cp.asarray(dt,dtype=dtype),args.dimensions,xp=cp))
        errors=[]
        gpu64_errors=[]
        prior_errors=[]
        for k,change in ((0,.01),(17,-.02),(35,.03)):
            change=np.dtype(dtype).type(change)
            old=cpu.dData
            new=old+cpu.cInduData[k]*float(change)
            baseline=f.likelihood_difference(old,new,cpu.a,args.dimensions)
            ratio=float(f.gpu_likelihood_difference(v.dData,v.cInduData[k],change,v.a,args.dimensions))
            gpu64=float(f.gpu_likelihood_difference(reference.dData,reference.cInduData[k],float(change),reference.a,args.dimensions))
            prior=(-float(change)*(cpu.dIndu-cpu.priorMean)[k]-.5*float(change)**2*cpu.cProposal[k,k])
            proposed=v.dIndu+v.cProposal[:,k]*change
            actual_prior=float(f.log_prior(proposed,v.priorMean,v.cInduInduInv)-f.log_prior(v.dIndu,v.priorMean,v.cInduInduInv))
            errors.append(abs(ratio-baseline));gpu64_errors.append(abs(ratio-gpu64));prior_errors.append(abs(actual_prior-prior))
        result=dict(dimensions=args.dimensions,count=args.count,stress=stress,full_likelihood_dtype=str(f.log_likelihood(v.dData,v.a,cp.asarray(dt,dtype=dtype),args.dimensions,xp=cp).dtype),dtype=dtype,condition=float(np.linalg.cond(cpu.cProposal)),
                    array_dtypes=sorted({str(x.dtype) for x in vars(v).values() if isinstance(x,cp.ndarray)}),
                    live_array_bytes=live,initialization_s=elapsed,inverse_residual=residual,
                    field_max_error=float(cp.max(cp.abs(v.dData-cp.asarray(cpu.dData)))),
                    minimum_diffusion=float(cp.min(v.dData)),full_likelihood_error=abs(ll-float(f.log_likelihood(cpu.dData,cpu.a,dt,args.dimensions))),
                    ratio_error_cpu=max(errors),ratio_error_gpu64=max(gpu64_errors),prior_delta_error=max(prior_errors))
        results.append(result);print(json.dumps(result),flush=True)
# Full original schedules on each predeclared seed and field, for both precisions.
for feature in (() if args.skip_inference else (False,True)):
    for seed in SEEDS:
        for dtype in ('float64','float32'):
            data=simulate(10000,seed,feature)
            cp.cuda.Device().synchronize();start=time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                v,samples,p=model.analyze(data,nIter=100,backend='gpu',dtype=dtype,seed=seed,
                    nInduX=6,nInduY=6,nFineX=6,nFineY=6,covLambda=.5,covL=1.2)
            cp.cuda.Device().synchronize()
            points=np.array([(x,y) for x in np.linspace(-1,1,3) for y in np.linspace(-1,1,3)])
            estimate=f.covMat(points,v.induCoordinates,v.covLambda,v.covL)@v.cInduInduInv@v.dIndu
            error=float(np.sqrt(np.mean(((estimate-truth(points,feature))/truth(points,feature))**2)))
            r=dict(feature=feature,seed=seed,dtype=dtype,seconds=time.perf_counter()-start,recovery_rms=error)
            results.append(r);print(json.dumps(r),flush=True)
failed=any(r.get('ratio_error_cpu',0)>RATIO_BUDGET or r.get('ratio_error_gpu64',0)>RATIO_BUDGET or r.get('prior_delta_error',0)>RATIO_BUDGET or r.get('recovery_rms',0)>.1 for r in results if r['dtype']=='float32')
results.append(dict(decision='reject float32; retain float64' if failed else 'float32 accuracy passed evaluated cases',ratio_budget=RATIO_BUDGET))
Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
print(results[-1],flush=True)
