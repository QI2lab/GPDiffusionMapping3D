"""Prepare spatial grids, regularized covariance and initial diffusion."""

from types import SimpleNamespace

import numba as nb
import numpy as np

from gp_math import covMat, log_likelihood, log_prior


def initialization(variables, data, backend="cpu", dtype="float64"):
    """Construct the spatial GP and initial diffusion state.

    Parameters
    ----------
    variables : dict
        Existing grid, covariance and regularization parameters.
    data : types.SimpleNamespace
        Prepared XY or ZYX observations with positive elapsed seconds.
    backend : {"cpu"}, optional
        Retained for existing CPU callers; other values are rejected.
    dtype : {"float64"}, optional
        Validated numerical precision.

    Returns
    -------
    types.SimpleNamespace
        Coordinates, fixed GP matrices and the initial diffusion values.
    """

    if backend != "cpu":
        raise ValueError("Only CPU execution is supported")
    if dtype != "float64":
        raise ValueError("Only float64 precision is supported")

    # Declare variables as instance of SimpleNamespace
    variables = SimpleNamespace(**variables)

    # Pull necessary variables
    trajectories = data.trajectories
    trajectoriesIndex = data.trajectoriesIndex
    epsilon = variables.epsilon
    dimensions = getattr(data, "dimensions", 2)
    if dimensions not in (2, 3):
        raise ValueError("Only XY 2D and ZYX 3D inference are supported")
    deltaT = data.delta_t if hasattr(data, "delta_t") else data.deltaT
    covL = variables.covL
    covLambda = variables.covLambda

    # Consecutive within-track pairs, without quadratic vstack copies.
    paired = trajectoriesIndex[:-1] == trajectoriesIndex[1:]
    dataCoordinates = np.ascontiguousarray(
        data.starts if hasattr(data, "starts") else trajectories[:-1][paired]
    )
    sampleCoordinates = np.ascontiguousarray(
        data.ends if hasattr(data, "ends") else trajectories[1:][paired]
    )

    # Find MLE assuming flat map
    diff = sampleCoordinates - dataCoordinates
    step_mle = np.sum(diff * diff, axis=1) / (2 * dimensions * deltaT)
    mle = np.mean(step_mle)

    # Estimate Hyperparameters if not chosen by user
    if covL is None:
        covL = np.max(np.ptp(trajectories, axis=0)) * 0.2
    if covLambda is None:
        covLambda = mle * 0.1

    induCoordinates, fineCoordinates, nIndu = spatial_grids(
        trajectories, variables, dimensions
    )

    # Set up initial sample and mean of prior
    dIndu = mle * np.ones(nIndu)
    priorMean = dIndu.copy()

    (
        cInduIndu, cInduData, cInduFine, cProposal,
        cInduInduInv, cDataIndu, cInduInduChol,
    ) = construct_covariances(
        induCoordinates, dataCoordinates, fineCoordinates, covLambda, covL,
        epsilon,
    )

    # Determine the mle at each individual data point
    a = np.asarray(np.sum(diff * diff, axis=1) / (4 * deltaT), dtype=dtype)
    dMleData = a * (2 / dimensions)


    dIndu, dData = smooth_initial_cpu(
        induCoordinates,
        dataCoordinates,
        covLambda,
        covL,
        dMleData,
        dIndu,
        nIndu,
        cDataIndu,
    )
    if bool(np.any(dData <= 0)) or bool(np.any(dIndu <= 0)):
        raise ValueError(
            "Initialization requires positive inducing and "
            "interpolated diffusion"
        )
    P = float(
        log_likelihood(
            dData, a, np.asarray(deltaT, dtype=dtype), dimensions
        )
        + log_prior(dIndu, priorMean, cInduInduInv)
    )
    variables.dimensions = dimensions
    variables.delta_t = np.broadcast_to(deltaT, (len(dataCoordinates),)).copy()
    variables.backend = backend
    variables.a = a

    print(
        f"The initial probability is {P} "
        f"and there are {nIndu} inducing points."
    )

    # Save all variable parameters
    variables.nIndu = nIndu
    variables.sampleCoordinates = sampleCoordinates
    variables.dataCoordinates = dataCoordinates
    variables.induCoordinates = induCoordinates
    variables.fineCoordinates = fineCoordinates
    variables.cProposal = cProposal
    variables.cInduIndu = cInduIndu
    variables.cInduData = cInduData
    variables.cInduFine = cInduFine
    variables.cInduInduChol = cInduInduChol
    variables.cInduInduInv = cInduInduInv
    variables.dIndu = dIndu
    variables.P = P
    variables.mle = mle
    variables.priorMean = priorMean
    variables.cDataIndu = cDataIndu
    variables.dData = dData
    variables.covLambda = covLambda
    variables.covL = covL

    return variables


def spatial_grids(trajectories, variables, dimensions):
    """Construct the existing XY or ZYX inducing and plotting grids."""
    nInduX, nInduY = variables.nInduX, variables.nInduY
    nFineX, nFineY = variables.nFineX, variables.nFineY
    minX = min(trajectories[:, 0])
    minY = min(trajectories[:, 1])
    maxX = max(trajectories[:, 0])
    maxY = max(trajectories[:, 1])
    # Define coordinates for Inducing points
    xIndu = np.linspace(minX, maxX, nInduX)
    yIndu = np.linspace(minY, maxY, nInduY)
    xTemp, yTemp = np.meshgrid(xIndu, yIndu)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    induCoordinates = np.vstack((X, Y)).T
    nIndu = len(induCoordinates)

    # Define coordinates for Fine points
    x = np.linspace(minX, maxX, nFineX)
    y = np.linspace(minY, maxY, nFineY)
    xTemp, yTemp = np.meshgrid(x, y)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    fineCoordinates = np.vstack((X, Y)).T

    if dimensions == 3:
        # Coordinates are explicitly ZYX. Timestamp is never a spatial axis.
        low = trajectories.min(axis=0)
        high = trajectories.max(axis=0)
        axes = [
            np.linspace(low[0], high[0], variables.nInduZ),
            np.linspace(low[1], high[1], nInduY),
            np.linspace(low[2], high[2], nInduX),
        ]
        induCoordinates = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        axes = [
            np.linspace(low[0], high[0], variables.nFineZ),
            np.linspace(low[1], high[1], nFineY),
            np.linspace(low[2], high[2], nFineX),
        ]
        fineCoordinates = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        nIndu = len(induCoordinates)

    return induCoordinates, fineCoordinates, nIndu


def construct_covariances(
    induCoordinates, dataCoordinates, fineCoordinates, covLambda, covL,
    epsilon,
):
    """Construct regularized covariance and interpolation matrices."""
    nIndu = len(induCoordinates)
    # Determine Covariance matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cProposal = cInduIndu + epsilon * np.mean(cInduIndu) * np.eye(
        nIndu, dtype=np.float64
    )
    cInduInduInv = np.linalg.inv(cProposal)
    cDataIndu = cInduData.T @ cInduInduInv
    cInduInduChol = np.linalg.cholesky(cProposal)

    return (
        cInduIndu, cInduData, cInduFine, cProposal,
        cInduInduInv, cDataIndu, cInduInduChol,
    )


@nb.jit(cache=True, nopython=True)
def smooth_initial_cpu(
    induCoordinates,
    dataCoordinates,
    covLambda,
    covL,
    dMleData,
    dIndu,
    nIndu,
    cDataIndu,
    smoother=1,
):

    # Set up matrix for interpolation
    """Apply the original CPU kernel-smoothing arithmetic."""
    cInduDataInterpolate = covMat(
        induCoordinates, dataCoordinates, covLambda, covL / smoother
    )

    # Find smoothing
    for i in range(nIndu):
        dIndu[i] = np.sum(dMleData * cInduDataInterpolate[i]) / np.sum(
            cInduDataInterpolate[i]
        )

    dData = cDataIndu @ dIndu

    return dIndu, dData
