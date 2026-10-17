# Scientific validation records

Current inference uses CPU float64. The [results](../RESULTS.md),
[CPU-only decision](../cpu-only-decision.md), and
[CPU optimization](../cpu-optimization.md) explain the implementation choices.
The [map-output guide](../map-export.md) records the latest export checks.

## Mathematical and physical evidence

- [Sampling mathematics](../sampling-repair-mathematics.md): posterior and proposal geometry.
- [Sampling repair](../sampling-repair-plan.md): authorized changes and convergence gates.
- [Burn-in investigation](../3d-burnin-investigation.md): uniform and smooth-field mixing.
- [Implementation stages](../stage-validation.md): the original staged validation timeline.
- [Refactoring](../refactoring-validation.md): mathematical and physical checks after extraction.
- [Test fixtures](../test-fixture-audit.md): fixture reuse and scientific test organization.
- [GitHub CI](../github-ci.md): short PR checks and requested convergence runs.

Historical records retain their original units, hardware, failed intermediate
experiments and commands. GPU results describe the retired implementation;
current inference is CPU-only. Older script paths now live under `tests/support/`.
For current instructions, use the [workflow](../workflow.md).

## Machine-readable measurements

JSON files retain their recorded numerical values and execution evidence.

- [cell batch validation](../cell-batch-validation.json)
- [cpu only scientific results](../cpu-only-scientific-results.json)
- [cpu optimization performance](../cpu-optimization-performance.json)
- [cpu optimized 2d performance](../cpu-optimized-2d-performance.json)
- [cpu optimized scientific results](../cpu-optimized-scientific-results.json)
- [cpu profile after](../cpu-profile-after.json)
- [cpu profile before](../cpu-profile-before.json)
- [cpu proposal microbenchmarks](../cpu-proposal-microbenchmarks.json)
- [final mathematics](../final-mathematics.json)
- [github ci results](../github-ci-results.json)
- [package validation](../package-validation.json)
- [sampling additive pilot](../sampling-additive-pilot.json)
- [sampling blocked development](../sampling-blocked-development.json)
- [sampling blocked8 development](../sampling-blocked8-development.json)
- [sampling blocked8 heldout](../sampling-blocked8-heldout.json)
- [sampling defaults after refactor initial](../sampling-defaults-after-refactor-initial.json)
- [sampling defaults after refactor](../sampling-defaults-after-refactor.json)
- [sampling defaults before refactor](../sampling-defaults-before-refactor.json)
- [sampling defaults initial check](../sampling-defaults-initial-check.json)
- [stage1 results](../stage1-results.json)
- [stage2 large results](../stage2-large-results.json)
- [stage2 results](../stage2-results.json)
- [stage4 burnin 10000](../stage4-burnin-10000.json)
- [stage4 cpu benchmark](../stage4-cpu-benchmark.json)
- [stage4 float32 results](../stage4-float32-results.json)
- [stage4 gpu benchmark](../stage4-gpu-benchmark.json)
- [stage4 paired sweep results](../stage4-paired-sweep-results.json)
- [stage4 sampling 1000](../stage4-sampling-1000.json)
- [stage4 sampling 5000](../stage4-sampling-5000.json)
- [stage5 2d one draw warmup](../stage5-2d-one-draw-warmup.json)
- [stage5 2d performance initial](../stage5-2d-performance-initial.json)
- [stage5 2d performance](../stage5-2d-performance.json)
- [stage5 scientific results](../stage5-scientific-results.json)
- [test fixture audit results](../test-fixture-audit-results.json)
