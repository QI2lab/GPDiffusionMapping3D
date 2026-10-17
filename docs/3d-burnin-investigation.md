# 3D burn-in investigation

> This record preserves the measurements and software state at the time.
> For current usage, see the [workflow](workflow.md) and [map outputs](map-export.md).

Question: Is the previous mixing failure specific to the smooth feature or
to a poorly posed simulation, and does longer burn-in help?

Protocol fixed before examining the new chains:
20,000 independent linked pairs; start positions uniform in [-3,3]^3 nm;
elapsed seconds uniform in [0.002,0.008]; data seed 41. Uniform truth D=1 or
smooth truth D=1+0.4*exp(-||r||^2/2), in nm^2/s. Same random start positions,
elapsed times and Gaussian innovations across cases; different physical
increments follow from the different diffusion fields. Both backends receive
identical data within each case. CPU sampler seed 41 and GPU sampler seed 42
are independent, not paired chains with forced matching decisions.

Five inducing points per axis, covLambda=0.5, covL=1.5, original regularization,
prior, interpolation, proposal distribution/correction and update order.
Both cases use the same grid construction. Their exact grid extents differ
slightly because the original grid covers observed endpoints as well as
starts; extents are reported explicitly rather than claimed identical.

Schedule: original 10,000 annealing sweeps and 200 stabilization/adaptation
sweeps; then 10,000 EXTRA discarded sweeps at temperature one with fixed
proposal widths; then 5,000 production sweeps at temperature one. Total
25,200 sweeps per chain versus 15,200 previously. This isolates extra
temperature-one burn-in without extending adaptation or changing the
annealing trajectory. Defaults remain unchanged outside explicit requests.

Four local processes run the two fields on CPU and actual CUDA concurrently.
BLAS/OMP thread counts are one. Concurrent durations are diagnostics, not
CPU/GPU speed benchmarks. The scalar CUDA probe synchronized successfully
on NVIDIA GeForce RTX 3090 and returned 10416.0 before these runs.

Command actually launched:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -u scripts/explore_3d_burnin.py --directory validation_artifacts/3d-burnin-10000 --output docs/stage4-burnin-10000.json
```

The first sandbox attempt failed because uv could not write its cache lock.
The same command/environment was approved and retried outside the sandbox.

## Independent controls and interpretation

Uniform scalar likelihood control: exact chi-square 99% interval contains the
known D=1. Its MLE is 1.0019406388, interval [0.9871976688,1.0170013535]. This
is an exact control for these independent uniform increments, not an oracle
for the 125-parameter GP posterior or a sampler replacement.

Unconstrained least-squares representation of known truth at observation
locations gives RMS 0.9681% uniform / 0.8293% smooth. At the 27 central
positions it gives 0.9676% / 0.3749%. Both fitted inducing/observation fields
are strictly positive. Thus gross inability of this grid to represent either
truth is not supported; these best-representation fits are not inference.
At least 40 starts lie within radius 0.5 nm of every central evaluation point.

Expected local posterior precision (likelihood Fisher plus GP precision)
condition numbers: inducing values ~90 for both cases; inverse coordinates
alpha=R^-1 D ~2.17e6 uniform / 1.97e6 smooth; prior-whitened coordinates
~1064 / 949. This local unconstrained approximation diagnoses geometry, not
global posterior convergence, multimodality, or proof of a sampler defect.
Diagonal scaling of the alpha precision still leaves condition numbers
~2.63e6 / 2.39e6, so unequal coordinate scales alone do not explain it.

Diagnostics: saved central-field traces, log posterior, split-half drift,
estimated ESS/autocorrelation, rank-normalized and folded split R-hat,
batch-length sensitivity, proposal epsilon and per-coordinate acceptance.
R-hat follows Vehtari et al., https://arxiv.org/abs/1903.08008.
Independent Gaussian control gave maximum rank split R-hat 1.00057; shifting
one chain by one standard deviation gave 1.16735 and flagged all coordinates.

Ten batches of 500 cannot automatically be treated as independent when
autocorrelation extends beyond that scale. The sensitivity table also uses
100,250,1000,2500-sweep batches. Large-batch nominal intervals with only two
batches are explicitly not a convergence pass. There are only two independent
chains per case, from the same initialization procedure with different RNG
seeds; this is exploratory and does not establish general convergence.

Raw traces/progress logs remain locally in ignored validation_artifacts/.
Numerical results are saved in [stage4-burnin-10000.json](stage4-burnin-10000.json). Stage 4 remains
incomplete until the scientific sampling concerns are resolved. No new
sampler, precision strategy, hyperparameter change or Stage 5 CLI is adopted.

## Preservation checks after adding optional burn-in

```bash
uv run --no-sync python -m pytest tests/cpu --strict-markers -q
```

30 passed in 365.91 s. This includes actual CPU 2D/3D mathematical and physical
cases; no CuPy backend is substituted for CPU.

```bash
GPD_REQUIRE_GPU=1 uv run --no-sync python -m pytest tests/gpu -k 'not independent_3d_chain_means_with_sampling_uncertainty' --strict-markers -q
```

16 passed, 1 deselected in 640.74 s; zero skips. The deselected case is the
previously executed, known failing long chain-mean comparison, not a GPU
availability skip. This investigation runs its separate longer-burn-in
comparison on actual CUDA and leaves the original gate unchanged.

The mathematics-only selections also passed: CPU 7 tests and GPU 6 tests.
All preservation checks used burn_in=0, retaining the original default schedule.

## Completed results

Case     Backend  MAP RMS  Minimum projected ESS  Max autocorr time  Zero-accept coefficients
Uniform  CPU      5.722%   6.624                  754.8 sweeps       3 / 125
Uniform  GPU      5.979%   5.880                  850.3 sweeps       2 / 125
Smooth   CPU      5.142%   5.855                  853.9 sweeps       3 / 125
Smooth   GPU      5.686%   5.023                  995.5 sweeps       1 / 125

Maximum rank-normalized split R-hat: uniform 1.78485, smooth 1.87365.
Maximum folded split R-hat: uniform 1.11797, smooth 1.08890.
27/27 uniform and 26/27 smooth positions exceed R-hat 1.01. Neither case
establishes adequate mixing. Original 500-sweep-batch comparisons fail at
5/27 uniform and 7/27 smooth points. With 1000-sweep batches the nominal
comparisons pass, but only five batches remain, MCSE has not stabilized, and
R-hat and near-frozen coefficients remain problematic. This is NOT a gate pass.

All four physical point-estimate recovery errors remain below the original
10% RMS target. The uniform scalar physical control passes. The available
representation fits are positive and accurate to <1% RMS. These findings do
not support attributing the problem only to a badly chosen smooth feature.
Uniform GP field inference also mixes poorly. This is a 125-parameter field
fit even when the ground truth is constant, not scalar constant-D inference.

The additional fixed-width burn-in did not resolve the problem. Compared with
the old smooth experiment, maximum rank split R-hat is 1.87365 versus 1.83926,
minimum projected ESS CPU/GPU is 5.86/5.02 versus 8.47/6.20. These are diagnostic
comparisons of two finite windows, not a claim that additional burn-in causes
worse convergence. The frozen-width phase cannot retune poor proposals.

Confirmed production behavior: zero accepted updates correspond to alpha
ranges below the conservative floating-point reconstruction resolution.
The mean conditional proposal standard deviations for those coefficients
are roughly 9,300 to 88,100 alpha units. The local Fisher approximations to
their conditional posterior standard deviations are about 0.091 to 0.327.
These approximations are not exact conditional quantiles, but the scale
mismatch is substantial. Epsilon is finite, not underflowed to zero. Across
the four chains acceptance fractions range from zero to 99.24%; aggregate
acceptance obscures poorly tuned individual coefficients. This identifies
proposal tuning as a concrete issue to investigate alongside coordinate
geometry, rather than claiming a demonstrated mathematical or CUDA defect.
The original cumulative-acceptance adaptation rules have not been changed.

Independent full-Gaussian recomputation at first, middle and final production
states agrees with saved log posteriors within 1.21e-9 for all four chains.
This does not support accumulated GPU likelihood drift as the explanation.

Concurrent elapsed times: uniform CPU/GPU 1635.90/2178.71 seconds; smooth
1655.72/2204.44 seconds. These include every schedule phase, compilation and
transfers in each worker, and overlap other correctness runs. They are NOT
performance benchmarks. Raw NPZ traces include proposal widths, acceptance,
initial/final alpha, production inducing values and projected field values.

Postprocessing actually run:

```bash
uv run --no-sync python scripts/explore_3d_burnin.py --directory validation_artifacts/3d-burnin-10000 --output docs/stage4-burnin-10000.json --summarize-only
uv run --no-sync python scripts/plot_3d_burnin.py validation_artifacts/3d-burnin-10000 docs/stage4-burnin-10000.json docs/figures/3d-burnin-10000.png
```

The plot shows central-field traces, running means and log posterior. Its
R-hat titles report maxima over all 27 sites. Dashed lines are simulation
truth, not known posterior means. No posterior credible-interval accuracy
claim is made from these unconverged traces. Stage 4 remains incomplete and
Stage 5 remains unstarted; no tolerance, seed, prior or sampler was changed.
