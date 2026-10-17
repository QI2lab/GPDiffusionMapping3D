"""Sequential compiled CPU Metropolis proposals for the exact GP target."""

import numba as nb
import numpy as np

from gp_math import indexShuffler, logNormpdf


@nb.jit(nopython=True, cache=True)
def diffusionPointSamplerSA_nb(
    nIndu,
    cInduIndu,
    cInduData,
    cInduInduInv,
    cDataIndu,
    deltaT,
    means,
    samples,
    data,
    chol,
    dInduOld,
    pOld,
    dDataOld,
    priorMean,
    covLambda,
    epsilon,
    temperature,
    diagnostics=None,
    additive=False,
):
    """Perform one sequential inverse-coordinate Metropolis sweep."""

    # Calculate probabilities of induced samples

    # Initialize inverse space
    alphaVect = cInduInduInv @ dInduOld

    # Counter for acceptances
    accCounter = np.zeros(np.shape(alphaVect))

    # Shuffle the index to sample through alpha vect randomly
    shuffledIndex = indexShuffler(nIndu)

    # Propose new dIndu by sampling random points in inverse space
    for pointIndex in shuffledIndex:
        # Propose new alpha point
        oldAlphaPoint = alphaVect[pointIndex]
        a = np.random.exponential(epsilon[pointIndex])
        proposal_scale = a if additive else a * oldAlphaPoint
        alphaDiff = proposal_scale * np.random.randn()
        newAlphaPoint = oldAlphaPoint + alphaDiff
        if diagnostics is not None:
            diagnostics[4, pointIndex] += abs(proposal_scale)

        # Incorporate new point into dInduNew and dDataNew
        dInduNew = dInduOld + cInduIndu[:, pointIndex] * alphaDiff
        dDataNew = dDataOld + cInduData[pointIndex, :] * alphaDiff

        # Make sure sampled diffusion values are all positive
        positive_inducing = np.all(dInduNew > 0)
        positive_data = np.all(dDataNew > 0)
        if diagnostics is not None:
            if not positive_inducing:
                diagnostics[0, pointIndex] += 1
            elif not positive_data:
                diagnostics[1, pointIndex] += 1
        if positive_data and positive_inducing:
            # Probability of old and new function
            pNew = cpu_log_posterior(
                dInduNew, dDataNew, priorMean, cInduInduInv,
                means, samples, deltaT,
            )

            # Compute acceptance probability from tempered distribution
            logPosteriorRatio = pNew - pOld
            logPropRatio = (
                0.0
                if additive
                else logNormpdf(diff=alphaDiff, sigma=a * newAlphaPoint)
                - logNormpdf(diff=alphaDiff, sigma=a * oldAlphaPoint)
            )

            # Apply simulated annealing by incorporating temperature
            if (logPosteriorRatio / temperature) + logPropRatio > np.log(
                np.random.rand()
            ):
                accCounter[pointIndex] = 1
                dInduOld = dInduNew
                dDataOld = dDataNew
                pOld = pNew
                alphaVect[pointIndex] = newAlphaPoint
                if diagnostics is not None:
                    diagnostics[3, pointIndex] += alphaDiff**2
            elif diagnostics is not None:
                diagnostics[2, pointIndex] += 1
    return dInduOld, pOld, dDataOld, accCounter


def diffusionPointSampler(variables, data):
    """Update one complete sequential Metropolis sweep.

    Parameters
    ----------
    variables : types.SimpleNamespace
        Current diffusion state, fixed GP matrices and proposal settings.
    data : types.SimpleNamespace
        Prepared observations for the original CPU reference path.

    Returns
    -------
    tuple
        Updated state and accepted proposals per coordinate or block.
    """

    if getattr(variables, "proposal", "relative") == "blocked":
        return sample_blocks(variables)

    # Necessary variables
    cInduIndu = variables.cProposal
    cInduData = variables.cInduData
    cInduInduInv = variables.cInduInduInv
    deltaT = variables.delta_t
    means = variables.dataCoordinates
    samples = variables.sampleCoordinates
    data = data.trajectories
    chol = variables.cInduInduChol
    dInduOld = variables.dIndu
    cDataIndu = variables.cDataIndu
    P = variables.P
    dData = variables.dData
    priorMean = variables.priorMean
    covLambda = variables.covLambda
    epsilon = variables.epsilon
    nIndu = variables.nIndu
    temperature = variables.temperature

    # Run numba version
    dIndu, P, dData, accCount = diffusionPointSamplerSA_nb(
        nIndu,
        cInduIndu,
        cInduData,
        cInduInduInv,
        cDataIndu,
        deltaT,
        means,
        samples,
        data,
        chol,
        dInduOld,
        P,
        dData,
        priorMean,
        covLambda,
        epsilon,
        temperature,
        getattr(variables, "proposal_stats", None),
        getattr(variables, "proposal", "relative") == "additive",
    )
    variables.dIndu = dIndu
    variables.P = P
    variables.dData = dData

    accRate = 100 * np.sum(accCount) / nIndu

    print(f"{accRate:.2f}%", end=" ")

    return variables, accCount


def prepare_block_proposals(v, block_size=16):
    """Freeze prior-whitened Fisher directions without changing the posterior.

    The positive-definite Fisher approximation is used only for a fixed
    symmetric proposal covariance. It is not an approximation to the target.
    """
    chol = v.cInduInduChol
    design = v.cDataIndu @ chol
    precision = np.eye(v.nIndu) + design.T @ (
        (v.dimensions / (2 * v.dData**2))[:, None] * design
    )
    factor = np.linalg.cholesky(precision)
    v.block_directions = chol @ np.linalg.solve(factor.T, np.eye(v.nIndu))
    v.block_basis = np.ascontiguousarray((v.cDataIndu @ v.block_directions).T)
    v.block_prior_directions = v.cInduInduInv @ v.block_directions
    v.block_prior_precision = v.block_directions.T @ v.block_prior_directions
    v.block_size = block_size
    v.nProposals = (v.nIndu + block_size - 1) // block_size


@nb.njit(cache=True)
def sample_blocks_nb(
    inducing,
    field,
    posterior,
    prior_mean,
    directions,
    basis,
    prior_directions,
    prior_precision,
    a,
    dimensions,
    block_size,
    epsilon,
    temperature,
    diagnostics,
):
    """Sequential symmetric block MH with exact Gaussian likelihood ratios."""
    accepted = np.zeros(len(epsilon))
    for block in indexShuffler(len(epsilon)):
        low = block * block_size
        high = min(len(inducing), low + block_size)
        change = epsilon[block] * np.random.randn(high - low)
        inducing_change = (
            np.ascontiguousarray(directions[:, low:high]) @ change
        )
        if diagnostics is not None:
            diagnostics[4, block] += epsilon[block]
        if np.any(inducing + inducing_change <= 0):
            if diagnostics is not None:
                diagnostics[0, block] += 1
            continue
        field_change = change @ basis[low:high]
        proposed = field + field_change
        if np.any(proposed <= 0):
            if diagnostics is not None:
                diagnostics[1, block] += 1
            continue
        difference = 0.0
        for j in range(len(field)):
            difference += -dimensions / 2 * np.log(proposed[j] / field[j]) - a[
                j
            ] * (1 / proposed[j] - 1 / field[j])
        difference -= change @ (
            np.ascontiguousarray(prior_directions[:, low:high].T)
            @ (inducing - prior_mean)
        )
        difference -= (
            0.5
            * change
            @ (
                np.ascontiguousarray(prior_precision[low:high, low:high])
                @ change
            )
        )
        if difference / temperature > np.log(np.random.rand()):
            inducing += inducing_change
            field = proposed
            posterior += difference
            accepted[block] = 1
            if diagnostics is not None:
                diagnostics[3, block] += change @ change
        elif diagnostics is not None:
            diagnostics[2, block] += 1
    return inducing, field, posterior, accepted


def sample_blocks(v):
    """Apply dependent block proposals in order using the compiled CPU sampler."""
    diagnostics = getattr(v, "proposal_stats", None)
    v.dIndu, v.dData, v.P, accepted = sample_blocks_nb(
        v.dIndu,
        v.dData,
        v.P,
        v.priorMean,
        v.block_directions,
        v.block_basis,
        v.block_prior_directions,
        v.block_prior_precision,
        v.a,
        v.dimensions,
        v.block_size,
        v.epsilon,
        v.temperature,
        diagnostics,
    )
    print(f"{100 * accepted.mean():.2f}%", end=" ")
    return v, accepted


@nb.njit(cache=True)
def cpu_log_posterior(
    dIndu_, dData_, priorMean, cInduInduInv, means, samples, deltaT,
):
    """Evaluate the original full-density CPU arithmetic in its order."""
    # Prior ignoring normalization
    diff = dIndu_ - priorMean
    prior = -0.5 * (diff.T @ (cInduInduInv @ diff))

    # Likelihood of the data
    lhood = 0
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            lhood += -0.5 * (samples[i, j] - means[i, j]) ** 2 / (
                2 * dData_[i] * deltaT[i]
            ) - 0.5 * np.log(2 * np.pi * 2 * dData_[i] * deltaT[i])
    prob = lhood + prior

    return prob
