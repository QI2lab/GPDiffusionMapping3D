import numpy as np
from types import SimpleNamespace
import functions
import copy
import time
import pickle

# NECESSARY VARIABLES TO RUN THE GIBBS SAMPLER
PARAMETERS = {
    # General Knowns
    "nInduZ": 5,
    "nFineZ": 10,
    "nInduX": 30,  # Number of x inducing points
    "nInduY": 30,  # Number of y grid points (total inducing points is nInduY*nInduX)
    "nFineX": 100,  # Fine grid points in x
    "nFineY": 100,  # Fine grid points in y (total fine points is nFineY*nFineX)
    "nIndu": 0,  # Number of inducing points after trimming
    # Knowns to be evaluated
    "dataCoordinates": None,  # All points of each trajectory except final location
    "sampleCoordinates": None,  # All points of each trajectory except initial location
    "induCoordinates": None,  # Coordinates of the inducing points
    "fineCoordinates": None,  # Coordinates of fine grid points
    "cInduIndu": None,  # Covariance matrix between inducing points
    "cInduData": None,  # Covariance matrix between inducing points and data points
    "cInduFine": None,  # Covariance matrix between inducing points and fine grid points
    "cInduInduChol": None,  # Cholesky decomposition of cInduIndu
    "cInduInduInv": None,  # Inverse of cInduIndu
    "cDataIndu": None,  # Product of cInduData*cInduInduInv
    # Variables
    "P": float,  # Probability of each sample
    "dIndu": None,  # Diffusion coefficient sample map at inducing points
    "dData": None,  # Interpolated diffusion from induCoordinates to dataCoordinates
    # Priors
    "covLambda": None,  # Coefficient of covariance square exponential kernel (1 only used if hyper parameters on specified)
    "covL": None,  # Lengths parameter of covariance square exponential kernel (20 only used if hyper parameters on specified)
    "mle": None,  # Prior on Inducing point MAP (set to MLE in init)
    "priorMean": None,  # Flat surface at MLE to be used as mean of GPP
    # Sampler parameters
    "epsilon": 1e-2,  # Perturbation parameter to keep matrix decomposition numerically stable and sample magnitude
    "temperature": 10,  # Temperature for simulated annealing will decay to one over time
}


# THIS FUNCTION RUNS THE GIBBS SAMPLER
def analyze(
    data,
    nIter=1000,
    backend="cpu",
    seed=None,
    dtype="float64",
    burn_in=0,
    **kwargs,
):
    """Run sequential Metropolis-within-Gibbs inference.

    Parameters
    ----------
    data : types.SimpleNamespace
        Prepared trajectory observations.
    nIter : int, optional
        Production sweeps, preceded by twice as many annealing sweeps.
    backend : {"cpu", "gpu"}, optional
        Explicit numerical backend. A requested GPU must be available.
    seed : int or None, optional
        Sampler random seed.
    dtype : {"float64", "float32"}, optional
        Numerical precision. Only float64 passed scientific validation.
    burn_in : int, optional
        Additional discarded temperature-one sweeps after adaptation, with
        fixed proposal widths. The default is zero extra sweeps.
    **kwargs
        Overrides of the existing model parameters.

    Returns
    -------
    map_variables : types.SimpleNamespace
        Highest-posterior production state and sampling diagnostics.
    dVect : numpy.ndarray
        Production inducing-value samples, one row per sweep.
    pVect : numpy.ndarray
        Unnormalized log posterior for each production sample.
    """

    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")

    if dtype == "float32":
        import warnings

        warnings.warn(
            "GPU float32 is experimental and failed acceptance-ratio validation; use float64 for inference",
            RuntimeWarning,
            stacklevel=2,
        )

    # Progress Marker
    print("Initialization Started")
    startTime = time.time()

    # Initialize data and variables and time them
    parameters = {**copy.deepcopy(PARAMETERS), **kwargs}
    if seed is not None:
        np.random.seed(seed)
        functions.seed_sampler(seed)
    if getattr(data, "dimensions", 2) == 3:
        if "nInduX" not in kwargs:
            parameters["nInduX"] = 5
        if "nInduY" not in kwargs:
            parameters["nInduY"] = 5
    variables = functions.initialization(
        parameters, data, backend=backend, dtype=dtype
    )
    endTime = time.time()

    # Progress Marker
    print("Initialization Successful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    # Get number of iterations
    burnIter = 2 * nIter
    stabilizeIter = 200

    # Redefine perturbation magnitude for samples
    variables.epsilon = np.ones(np.shape(variables.dIndu))
    startTime = time.time()

    # Initial temp and cooling rate for targeted annealing
    initTemp = 7.5
    coolRate = np.log(initTemp) / burnIter

    ## BURN IN 1 ##
    # Burn in super aggressively but do not save the samples,
    # updating proposal size to maintain healthy acceptance rate
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(burnIter):
        variables.temperature = functions.expCooling(i, initTemp, coolRate)
        print(f"Annealing Iteration {i + 1}/{burnIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        accVectNorm = 100 * accVect / (i + 1)
        variables.epsilon = np.where(
            accVectNorm > 40,
            variables.epsilon * 1.5,
            np.where(
                accVectNorm < 20, variables.epsilon * 0.5, variables.epsilon
            ),
        )

        print(f"({time.time() - t:.3f}s)")

    ## Stabilize proposal Magnitude ##
    # Set temperature back to 1 and acceptance to 0, and iterate
    # a few times to find ideal magnitude of proposal to maintain healthy acceptance
    variables.temperature = 1
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(stabilizeIter):
        print(f"Stabilization Iteration {i + 1}/{stabilizeIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        if i > 10:
            accVectNorm = 100 * accVect / (i + 1)
            variables.epsilon = np.where(
                accVectNorm > 27.5,
                variables.epsilon * 1.25,
                np.where(
                    accVectNorm < 22.5,
                    variables.epsilon * 0.75,
                    variables.epsilon,
                ),
            )

        print(f"({time.time() - t:.3f}s)")

    # Optional additional burn-in does not alter the adaptation or annealing.
    for i in range(burn_in):
        print(f"Discarded Burn-in Iteration {i + 1}/{burn_in}", end=" ")
        variables, _ = functions.diffusionPointSampler(variables, data)
        print()

    def host_array(value):
        return value.get() if backend == "gpu" else np.array(value, copy=True)

    initial_inducing = host_array(variables.dIndu)
    initial_alpha = host_array(variables.cInduInduInv @ variables.dIndu)
    production_acceptance = np.zeros(variables.nIndu)

    ## MCMC SAMPLING ##

    # Initialize outputs
    dVect = np.zeros((nIter, variables.nIndu))
    pVect = np.zeros(nIter)
    map_variables = copy.deepcopy(variables)
    map_variables.P = -np.inf

    # Gibbs Sampler
    for i in range(nIter):
        print(f"MCMC Iteration {i + 1}/{nIter}", end=" ")
        t = time.time()

        # Sample diffusion map
        variables, accCount = functions.diffusionPointSampler(variables, data)
        accVect += accCount
        production_acceptance += accCount

        # Save the sample
        dVect[i] = (
            variables.dIndu.get() if backend == "gpu" else variables.dIndu
        )
        pVect[i] = variables.P
        if variables.P >= map_variables.P:
            map_variables = copy.deepcopy(variables)

        print(f"({time.time() - t:.3f}s)")

    endTime = time.time()

    print(
        str(nIter) + " iterations in " + str(endTime - startTime) + " seconds."
    )

    map_variables.sampling_metadata = {
        "annealing_sweeps": burnIter,
        "stabilization_sweeps": stabilizeIter,
        "discarded_fixed_width_sweeps": burn_in,
        "production_sweeps": nIter,
        "epsilon": np.array(variables.epsilon, copy=True),
        "initial_inducing": initial_inducing,
        "initial_alpha": initial_alpha,
        "final_alpha": host_array(variables.cInduInduInv @ variables.dIndu),
        "acceptance_fraction": production_acceptance / nIter,
    }

    if backend == "gpu":
        import cupy as cp

        cp.cuda.Device().synchronize()
        for key, value in vars(map_variables).items():
            if isinstance(value, cp.ndarray):
                setattr(map_variables, key, cp.asnumpy(value))
        for key in (
            "dataCoordinates",
            "sampleCoordinates",
            "induCoordinates",
            "fineCoordinates",
        ):
            setattr(
                map_variables,
                key,
                getattr(map_variables, key) + map_variables.coordinate_origin,
            )
        map_variables.coordinate_origin = np.zeros(map_variables.dimensions)
    return map_variables, dVect, pVect
