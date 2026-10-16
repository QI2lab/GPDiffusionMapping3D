# Necessary Imports
import numpy as np
from types import SimpleNamespace
from scipy import stats
import numba as nb

# Set seed
np.random.seed(42)

# Function that determines temperature for simulated annealing
def expCooling(iteration, initialTemp, coolRate):
    """
    Determine temperature for simulated annealing.

    Parameters:
    - iteration: Current iteration number.
    - initialTemp: Initial temperature.
    - coolRate: Cooling rate.

    Returns:
    - Temperature for the current iteration.
    """
    return initialTemp * np.exp(-coolRate * iteration)

# Function that calculates unnormalized logpdf of Normal distribution assuming n=1
@nb.njit(cache=True)
def logNormpdf(diff, sigma):
    """
    Calculate the unnormalized log probability density function (pdf) of a normal distribution.

    Parameters:
    - diff: Difference.
    - sigma: Standard deviation.

    Returns:
    - Unnormalized log pdf value.
    """
    return -np.log(np.abs(sigma)) - 0.5 * (diff / sigma) ** 2

# Function that randomizes index sequence for point sampler
@nb.njit(cache=True)
def indexShuffler(length):
    """
    Randomly shuffle the index sequence for sampling points.

    Parameters:
    - length: Length of the index sequence.

    Returns:
    - Shuffled index sequence.
    """
    indices = np.arange(length)
    for i in range(length):
        j = int(np.random.random() * (length - i)) + i
        indices[i], indices[j] = indices[j], indices[i]
    return indices

# Create a covariance matrix based on data at hand
@nb.njit(cache=True)
def covMat(coordinates1, coordinates2, covLambda, covL):
    """
    Create a covariance matrix based on given coordinates and parameters.

    Parameters:
    - coordinates1: Coordinates 1.
    - coordinates2: Coordinates 2.
    - covLambda: Standard Deviation Coefficient.
    - covL: Covariance Length Scale.

    Returns:
    - Covariance matrix.
    """
    C = np.zeros((len(coordinates1), len(coordinates2)))
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            dist_squared = 0.0
            for axis in range(coordinates1.shape[1]):
                dist_squared += (coordinates1[i, axis] - coordinates2[j, axis]) ** 2
            dist = np.sqrt(dist_squared)
            C[i, j] = (covLambda ** 2) * (np.exp(((-0.5) * ((dist) ** 2)) / (covL ** 2)))
    return C

# Initialize sampler parameters
def initialization(variables, data, backend="cpu", dtype="float64"):
    """
    Initialize parameters for the sampler.

    Parameters:
    - variables: Variables to initialize.
    - data: Data needed for initialization.

    Returns:
    - Initialized variables.
    """

    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    if dtype not in ("float64", "float32") or (backend == "cpu" and dtype != "float64"):
        raise ValueError("CPU requires float64; GPU evaluation supports float64 or float32")
    xp = np
    if backend == "gpu":
        import cupy as cp
        cp.cuda.Device().use()
        xp = cp

    # Declare variables as instance of SimpleNamespace
    variables = SimpleNamespace(**variables)

    # Pull necessary variables
    trajectories = data.trajectories
    nData = data.nData
    trajectoriesIndex = data.trajectoriesIndex
    nInduX = variables.nInduX
    nInduY = variables.nInduY
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    epsilon = variables.epsilon
    dimensions = getattr(data, "dimensions", 2)
    if dimensions not in (2, 3):
        raise ValueError("Only XY 2D and ZYX 3D inference are supported")
    deltaT = data.delta_t if hasattr(data, "delta_t") else data.deltaT
    dataX = trajectories[:, 0]
    dataY = trajectories[:, 1]
    minX = min(dataX)
    minY = min(dataY)
    maxX = max(dataX)
    maxY = max(dataY)
    covL = variables.covL
    covLambda = variables.covLambda

    # Identical consecutive within-track pairs, without quadratic vstack copies.
    paired = trajectoriesIndex[:-1] == trajectoriesIndex[1:]
    dataCoordinates = np.ascontiguousarray(data.starts if hasattr(data, "starts") else trajectories[:-1][paired])
    sampleCoordinates = np.ascontiguousarray(data.ends if hasattr(data, "ends") else trajectories[1:][paired])

    # Find MLE assuming flat map
    diff = sampleCoordinates - dataCoordinates
    step_mle = np.sum(diff * diff, axis=1) / (2 * dimensions * deltaT)
    mle = np.mean(step_mle)

    # Estimate Hyperparameters if not chosen by user
    if covL == None:
        covL = np.max(np.ptp(trajectories, axis=0)) * 0.2
    if covLambda == None:
        covLambda = mle * 0.1

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
        axes = [np.linspace(low[0], high[0], variables.nInduZ),
                np.linspace(low[1], high[1], nInduY),
                np.linspace(low[2], high[2], nInduX)]
        induCoordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        axes = [np.linspace(low[0], high[0], variables.nFineZ),
                np.linspace(low[1], high[1], nFineY),
                np.linspace(low[2], high[2], nFineX)]
        fineCoordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        nIndu = len(induCoordinates)

    # Set up initial sample and mean of prior
    dIndu = mle * np.ones(nIndu)
    priorMean = dIndu.copy()

    diff64 = sampleCoordinates - dataCoordinates
    variables.coordinate_origin = np.zeros(dimensions)
    if dtype == "float32":
        origin = dataCoordinates.min(axis=0)
        variables.coordinate_origin = origin
        dataCoordinates = dataCoordinates - origin
        sampleCoordinates = sampleCoordinates - origin
        induCoordinates = induCoordinates - origin
        fineCoordinates = fineCoordinates - origin

    # Transfer numerical arrays once; retain proposal work on the device.
    if backend == "gpu":
        dataCoordinates = xp.asarray(dataCoordinates, dtype=dtype)
        sampleCoordinates = xp.asarray(sampleCoordinates, dtype=dtype)
        induCoordinates = xp.asarray(induCoordinates, dtype=dtype)
        fineCoordinates = xp.asarray(fineCoordinates, dtype=dtype)
        dIndu = xp.asarray(dIndu, dtype=dtype)
        priorMean = xp.asarray(priorMean, dtype=dtype)
    kernel = covMat if backend == "cpu" else covMat_gpu

    # Determine Covariance matrices
    cInduIndu = kernel(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = kernel(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = kernel(induCoordinates, fineCoordinates, covLambda, covL)
    cProposal = cInduIndu + epsilon * xp.mean(cInduIndu) * xp.eye(nIndu, dtype=dtype)
    cInduInduInv = xp.linalg.inv(cProposal)
    cDataIndu = cInduData.T @ cInduInduInv
    cInduInduChol = xp.linalg.cholesky(cProposal)

    # Determine the mle at each individual data point
    a = xp.asarray(np.sum(diff64 * diff64, axis=1) / (4 * deltaT), dtype=dtype)
    dMleData = a * (2 / dimensions)

    @nb.jit(cache=True, nopython=True)
    def compute_vector(induCoordinates, dataCoordinates, covLambda, covL, dMleData, dIndu, nIndu, cDataIndu, smoother=1):

        # Set up matrix for interpolation
        cInduDataInterpolate = covMat(induCoordinates, dataCoordinates, covLambda, covL / smoother)

        # Find smoothing
        for i in range(nIndu):
            dIndu[i] = np.sum(dMleData * cInduDataInterpolate[i]) / np.sum(cInduDataInterpolate[i])

        dData = cDataIndu @ dIndu

        return dIndu, dData

    def calculate_log_posterior(dataCoordinates, sampleCoordinates, dData, deltaT, dIndu, priorMean, cInduInduInv):
        # Likelihood of the data
        lhood = np.sum(
            stats.norm.logpdf(
                sampleCoordinates,
                loc=dataCoordinates,
                scale=np.sqrt(2 * np.vstack((dData, dData)).T * deltaT)
            )
        )

        # Prior of the Data ignoring normalization
        diff = dIndu - priorMean
        prior = -0.5 * (diff.T @ (cInduInduInv @ diff))
        pTemp = lhood + prior

        return pTemp

    P = -np.inf  # Set to negative infinity initially or any other large negative value to ensure the loop runs at least once

    if backend == "cpu":
        dIndu, dData = compute_vector(
            induCoordinates, dataCoordinates, covLambda, covL, dMleData,
            dIndu, nIndu, cDataIndu)
    else:
        dIndu = (cInduData @ dMleData) / xp.sum(cInduData, axis=1)
        dData = cDataIndu @ dIndu
    if bool(xp.any(dData <= 0)) or bool(xp.any(dIndu <= 0)):
        raise ValueError("Initialization requires positive inducing and interpolated diffusion")
    P = float(log_likelihood(dData, a, xp.asarray(deltaT, dtype=dtype), dimensions, xp)
              + log_prior(dIndu, priorMean, cInduInduInv))
    variables.dimensions = dimensions
    variables.delta_t = np.broadcast_to(deltaT, (len(dataCoordinates),)).copy()
    variables.backend = backend
    variables.a = a

    print(f"The initial probability is {P} and there are {nIndu} inducing points.")

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

@nb.jit(nopython=True, cache=True)
def diffusionPointSamplerSA_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol,
                               dInduOld, pOld, dDataOld, priorMean, covLambda, epsilon, temperature):
    """
    Numba-compiled function for sampling in the inverse space.

    Parameters:
    - nIndu: Number of inducing points.
    - cInduIndu: Autocovariance matrix.
    - cInduData: Covariance data->indu.
    - cInduInduInv: Inverse of autocovariance matrix.
    - cDataIndu: Data covariance.
    - deltaT: Time interval.
    - means: Mean values.
    - samples: Samples data.
    - data: Data.
    - chol: Cholesky decomposition.
    - dInduOld: Old inducing points.
    - pOld: Old probability.
    - dDataOld: Old data.
    - priorMean: Prior mean.
    - covLambda: Covariance lambda.
    - epsilon: Epsilon parameter.
    - temperature: Temperature.

    Returns:
    - Sampled inducing points, probability, sampled Diffusion at Data points and acceptance rate.
    """

    # Calculate probabilities of induced samples
    def probability(dIndu_, dData_):
        # Prior ignoring normalization
        diff = dIndu_ - priorMean
        prior = -0.5 * (diff.T @ (cInduInduInv @ diff))

        # Likelihood of the data
        lhood = 0
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                lhood += (
                    -0.5 * (samples[i, j] - means[i, j]) ** 2 / (2 * dData_[i] * deltaT[i])
                    - 0.5 * np.log(2 * np.pi * 2 * dData_[i] * deltaT[i])
                )
        prob = lhood + prior

        return prob

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
        alphaDiff = a * oldAlphaPoint * np.random.randn()
        newAlphaPoint = oldAlphaPoint + alphaDiff

        # Incorporate new point into dInduNew and dDataNew
        dInduNew = dInduOld + cInduIndu[:, pointIndex] * alphaDiff
        dDataNew = dDataOld + cInduData[pointIndex, :] * alphaDiff

        # Make sure sampled diffusion values are all positive
        if np.all(dDataNew > 0) and np.all(dInduNew > 0):

            # Probability of old and new function
            pOld = pOld
            pNew = probability(dInduNew, dDataNew)

            # Compute acceptance probability from tempered distribution
            logPosteriorRatio = pNew - pOld
            logPropRatio = (logNormpdf(diff = alphaDiff, sigma = a * newAlphaPoint) 
            - logNormpdf(diff = alphaDiff, sigma = a * oldAlphaPoint))

            # Apply simulated annealing by incorporating temperature
            if (logPosteriorRatio / temperature) + logPropRatio > np.log(np.random.rand()):
                accCounter[pointIndex] = 1
                dInduOld = dInduNew
                dDataOld = dDataNew
                pOld = pNew
                alphaVect[pointIndex] = newAlphaPoint
    return dInduOld, pOld, dDataOld, accCounter


# This function is a Metropolis sampler that samples from inverse Gaussian process space
def diffusionPointSampler(variables, data):
    """
    Metropolis sampler that samples from the inverse Gaussian process space.

    Parameters:
    - variables: Variables for sampling.
    - data: Data needed for sampling.

    Returns:
    - Sampled variables and acceptance count.
    """

    if getattr(variables, "backend", "cpu") == "gpu":
        return diffusionPointSampler_gpu(variables)

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
    dIndu, P, dData, accCount = diffusionPointSamplerSA_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu,
                                                            deltaT, means, samples, data, chol, dInduOld, P,
                                                            dData, priorMean, covLambda, epsilon, temperature)
    variables.dIndu = dIndu
    variables.P = P
    variables.dData = dData

    accRate = 100 * np.sum(accCount) / nIndu

    print(f"{accRate:.2f}%", end=" ")

    return variables, accCount



def covMat_gpu(coordinates1, coordinates2, covLambda, covL):
    """Evaluate the existing spatial RBF on CUDA using CuPy built-ins."""
    import cupy as cp

    squared = cp.zeros((len(coordinates1), len(coordinates2)), dtype=coordinates1.dtype)
    for axis in range(coordinates1.shape[1]):
        squared += (coordinates1[:, axis, None] - coordinates2[None, :, axis])**2
    return covLambda**2 * cp.exp(-.5 * squared / covL**2)


def log_likelihood(diffusion, a, delta_t, dimensions=2, xp=np):
    """Evaluate the full Gaussian transition density, including constants."""
    if diffusion.dtype == np.dtype("float32"):
        delta_t = xp.asarray(delta_t, dtype=diffusion.dtype)
    if bool(xp.any(diffusion <= 0)):
        return -xp.inf
    return -xp.sum(dimensions / 2 * xp.log(diffusion) + a / diffusion
                   + dimensions / 2 * xp.log(4 * xp.pi * delta_t))


def log_prior(diffusion, mean, inverse):
    """Evaluate the original unnormalized direct GP prior."""
    difference = diffusion - mean
    return -.5 * (difference @ (inverse @ difference))


def likelihood_difference(old, new, a, dimensions=2, xp=np):
    """Evaluate a proposal ratio directly without subtracting total densities."""
    if bool(xp.any(new <= 0)):
        return -xp.inf
    return xp.sum(-dimensions / 2 * xp.log(new / old)
                  - a * (1 / new - 1 / old))


def proposal_log_ratio(change, scale, old, new):
    """Return the original conditional proposal-density correction."""
    if old == 0 or new == 0:
        return -np.inf
    return float(logNormpdf(change, scale * new)
                 - logNormpdf(change, scale * old))


_proposal_reduction = None
_proposal_reduction32 = None


def gpu_likelihood_difference(old, column, change, a, dimensions=2):
    """Fuse the measured proposal likelihood and positivity reduction."""
    import cupy as cp

    global _proposal_reduction, _proposal_reduction32
    if old.dtype == cp.float32:
        if _proposal_reduction32 is None:
            _proposal_reduction32 = cp.ReductionKernel(
                "float32 old, float32 column, float32 change, float32 a, float32 half_dimension",
                "float32 result",
                "old + column * change > 0 ? "
                "-half_dimension * logf((old + column * change) / old) "
                "-a * (1.0f / (old + column * change) - 1.0f / old) : -(1.0f/0.0f)",
                "a + b", "result = a", "0", "gpd_proposal_float32_spatial")
        return _proposal_reduction32(old, column, change, a, dimensions / 2)
    if _proposal_reduction is None:
        _proposal_reduction = cp.ReductionKernel(
            "float64 old, float64 column, float64 change, float64 a, float64 half_dimension",
            "float64 result",
            "old + column * change > 0 ? "
            "-half_dimension * log((old + column * change) / old) "
            "-a * (1.0 / (old + column * change) - 1.0 / old) : -(1.0/0.0)",
            "a + b", "result = a", "0", "gpd_proposal_float64_spatial")
    return _proposal_reduction(old, column, change, a, dimensions / 2)


def diffusionPointSampler_gpu(variables):
    """Sequential Metropolis proposals with resident CuPy numerical arrays."""
    import cupy as cp

    v = variables
    alpha = cp.asnumpy(v.cInduInduInv @ v.dIndu)
    inducing_host = cp.asnumpy(v.dIndu)
    mean_host = cp.asnumpy(v.priorMean)
    proposal_host = cp.asnumpy(v.cProposal)
    accepted = np.zeros(v.nIndu)
    for k in indexShuffler(v.nIndu):
        old_alpha = alpha[k]
        scale = np.random.exponential(v.epsilon[k])
        change = scale * old_alpha * np.random.randn()
        if v.dData.dtype == cp.float32:
            change = np.float32(change)
        new_alpha = old_alpha + change
        inducing = inducing_host + proposal_host[:, k] * change
        if np.all(inducing > 0):
            delta = float(gpu_likelihood_difference(v.dData, v.cInduData[k], change, v.a, v.dimensions))
            # Exact quadratic-prior difference for a column of R, R^-1 R_k=e_k.
            delta += (-change * (inducing_host - mean_host)[k]
                      -.5 * change**2 * proposal_host[k, k])
            correction = proposal_log_ratio(change, scale, old_alpha, new_alpha)
            if delta / v.temperature + correction > np.log(np.random.rand()):
                accepted[k] = 1
                v.dIndu += v.cProposal[:, k] * change
                v.dData += v.cInduData[k] * change
                inducing_host = inducing
                v.P += delta
                alpha[k] = new_alpha
    print(f"{100 * accepted.mean():.2f}%", end=" ")
    return v, accepted


@nb.njit(cache=True)
def seed_sampler(seed):
    """Set the CPU sampler RNG for independently seeded inference runs."""
    np.random.seed(seed)
