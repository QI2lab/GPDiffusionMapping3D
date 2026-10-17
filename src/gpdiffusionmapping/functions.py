"""Compatibility entry points for the validated numerical functions."""

import numpy as np

from .gp_math import (
    expCooling,
    logNormpdf,
    indexShuffler,
    covMat,
    log_likelihood,
    log_prior,
    likelihood_difference,
    proposal_log_ratio,
    seed_sampler,
)
from .gp_initialization import initialization
from .metropolis import (
    diffusionPointSamplerSA_nb,
    diffusionPointSampler,
    prepare_block_proposals,
    sample_blocks_nb,
    sample_blocks,
)

# Preserve the original import-time NumPy seed. Explicit inference seeds override it.
np.random.seed(42)
