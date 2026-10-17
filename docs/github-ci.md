# GitHub CI

The short scientific suite runs on pull requests and pushes to `main`/`master`.
Feature-branch pushes do not trigger a duplicate run. Manual runs are available
through GitHub Actions.

| Check | Coverage |
| --- | --- |
| Unit | Independent mathematical identities and proposal calculations |
| Posterior | Sampler results against numerical quadrature |
| Physics | Brownian laws and 2D/3D field recovery |
| Files | Per-cell CSV/Parquet inference and calibrated OME-TIFF export |

The four checks cover 56 cases. Deprecation and pending-deprecation warnings
fail tests. Results and available scientific diagnostics are retained as
GitHub Actions artifacts for seven days.

## Request full convergence

Create the repository label once: open
[Labels](https://github.com/dpshepherd/GPDiffusionMapping/labels), click
**New label**, enter `run-convergence`, and click **Create label**.

Add `run-convergence` to an open, non-draft PR ready to merge. This requests
the two full 3D convergence cases, using four independent chains and 20,000
production sweeps per chain for uniform and smooth fields.

There is no nightly run. New commits do not rerun convergence automatically;
remove and re-add the label to request a run for the latest revision. Verify
both convergence results belong to the revision being merged.
