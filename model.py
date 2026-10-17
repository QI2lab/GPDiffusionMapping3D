"""Inference setup, warmup, production sampling and result collection."""

import copy
import time

import numpy as np

import functions

# Existing GP and sampler parameter defaults
PARAMETERS = {
    # General Knowns
    "nInduZ": 5,
    "nFineZ": 10,
    "nInduX": 30,  # Number of x inducing points
    "nInduY": 30,  # Number of y inducing points
    "nFineX": 100,  # Fine grid points in x
    "nFineY": 100,  # Number of fine y grid points
    "nIndu": 0,  # Number of inducing points after trimming
    # Knowns to be evaluated
    "dataCoordinates": None,  # Transition start locations
    "sampleCoordinates": None,  # Transition end locations
    "induCoordinates": None,  # Coordinates of the inducing points
    "fineCoordinates": None,  # Coordinates of fine grid points
    "cInduIndu": None,  # Covariance matrix between inducing points
    "cInduData": None,  # Covariance between inducing and observation points
    "cInduFine": None,  # Covariance between inducing and fine grid points
    "cInduInduChol": None,  # Cholesky decomposition of cInduIndu
    "cInduInduInv": None,  # Inverse of cInduIndu
    "cDataIndu": None,  # Product of cInduData*cInduInduInv
    # Variables
    "P": float,  # Probability of each sample
    "dIndu": None,  # Diffusion coefficient sample map at inducing points
    "dData": None,  # Diffusion interpolated at observations
    # Priors
    "covLambda": None,  # Kernel standard deviation; estimated if unspecified
    "covL": None,  # Kernel length; estimated if unspecified
    "mle": None,  # Prior on Inducing point MAP (set to MLE in init)
    "priorMean": None,  # Flat surface at MLE to be used as mean of GPP
    # Sampler parameters
    "epsilon": 1e-2,  # Original covariance regularization coefficient
    "temperature": 10,  # Annealing temperature; production uses one
}


def analyze(
    data,
    nIter=1000,
    backend="cpu",
    seed=None,
    dtype="float64",
    burn_in=None,
    adaptation=None,
    annealing_sweeps=None,
    adaptation_sweeps=None,
    diagnostics=False,
    initial_inducing=None,
    proposal=None,
    block_size=8,
    **kwargs,
):
    """Run sequential Metropolis-within-Gibbs inference.

    Parameters
    ----------
    data : types.SimpleNamespace
        Prepared trajectory observations.
    nIter : int, optional
        Production sweeps. Multiple chains and convergence diagnostics remain
        necessary; the 3D validation used 20000 sweeps per chain.
    backend : {"cpu", "gpu"}, optional
        Explicit numerical backend. A requested GPU must be available.
    seed : int or None, optional
        Sampler random seed.
    dtype : {"float64", "float32"}, optional
        Numerical precision. Only float64 passed scientific validation.
    burn_in : int or None, optional
        Additional discarded temperature-one sweeps after adaptation, with
        fixed widths. Defaults to 1000 for blocked proposals, else zero.
    adaptation : {"legacy", "diminishing"} or None, optional
        Defaults to legacy cumulative adaptation for relative proposals and
        recent-window log-width tuning for additive or blocked proposals.
    annealing_sweeps, adaptation_sweeps : int or None, optional
        Explicit warmup lengths. Blocked proposals default to 2000 annealing
        sweeps; other proposals use twice nIter. Adaptation defaults to 200
        sweeps for legacy tuning or 4000 for diminishing tuning.
    diagnostics : bool, optional
        Record warmup widths, rejections and accepted displacements.
    initial_inducing : numpy.ndarray or None, optional
        Positive initial inducing field for independent-chain investigation.
    proposal : {"relative", "additive", "blocked"} or None, optional
        Legacy state-dependent or symmetric additive inverse-coordinate moves.
        The blocked option uses fixed prior-whitened Fisher directions.
        Defaults to blocked for 3D and the original relative proposal for 2D.
    block_size : int, optional
        Number of coordinates in a sequential symmetric block proposal.
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

    if proposal is None:
        proposal = (
            "blocked" if getattr(data, "dimensions", 2) == 3 else "relative"
        )
    if adaptation is None:
        adaptation = "legacy" if proposal == "relative" else "diminishing"
    if burn_in is None:
        burn_in = 1000 if proposal == "blocked" else 0
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if adaptation not in ("legacy", "diminishing"):
        raise ValueError("adaptation must be legacy or diminishing")
    if proposal not in ("relative", "additive", "blocked"):
        raise ValueError("proposal must be relative, additive or blocked")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if proposal == "blocked" and dtype != "float64":
        raise ValueError(
            "Blocked proposals require validated float64 precision"
        )
    burnIter = (
        (2000 if proposal == "blocked" else 2 * nIter)
        if annealing_sweeps is None
        else annealing_sweeps
    )
    stabilizeIter = (
        (200 if adaptation == "legacy" else 4000)
        if adaptation_sweeps is None
        else adaptation_sweeps
    )
    for length in (nIter, burnIter, stabilizeIter):
        if not isinstance(length, (int, np.integer)) or length < 0:
            raise ValueError("Iteration counts must be nonnegative integers")
    if nIter == 0:
        raise ValueError("Production sweeps must be positive")

    if dtype == "float32":
        import warnings

        warnings.warn(
            "GPU float32 is experimental and failed acceptance-ratio "
            "validation; use float64 for inference",
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
    variables.proposal = proposal
    if initial_inducing is not None:
        xp = np
        if backend == "gpu":
            import cupy as cp

            xp = cp
        initial = xp.asarray(initial_inducing, dtype=dtype)
        if initial.shape != variables.dIndu.shape or not bool(
            xp.all(xp.isfinite(initial))
        ):
            raise ValueError(
                "Initial inducing field must match the finite grid"
            )
        field = variables.cDataIndu @ initial
        if bool(xp.any(initial <= 0)) or bool(xp.any(field <= 0)):
            raise ValueError(
                "Initial inducing and observed diffusion must be positive"
            )
        variables.dIndu = initial.copy()
        variables.dData = field
        variables.P = float(
            functions.log_likelihood(
                field,
                variables.a,
                xp.asarray(variables.delta_t),
                variables.dimensions,
                xp,
            )
            + functions.log_prior(
                initial, variables.priorMean, variables.cInduInduInv
            )
        )
    variables.nProposals = variables.nIndu
    if proposal == "blocked":
        functions.prepare_block_proposals(variables, block_size)
    if diagnostics:
        variables.proposal_stats = np.zeros((5, variables.nProposals))
    endTime = time.time()

    # Progress Marker
    print("Initialization Successful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    # Get number of iterations

    startTime = time.time()
    warmup_trace = run_warmup(
        variables, data, burnIter, stabilizeIter, burn_in,
        adaptation, diagnostics,
    )

    def host_array(value):
        return value.get() if backend == "gpu" else np.array(value, copy=True)

    initial_inducing = host_array(variables.dIndu)
    initial_alpha = host_array(variables.cInduInduInv @ variables.dIndu)
    if diagnostics:
        warmup_stats = variables.proposal_stats.copy()
        variables.proposal_stats[:] = 0

    map_variables, dVect, pVect, production_acceptance = run_production(
        variables, data, nIter, backend
    )

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
        "adaptation": adaptation,
        "proposal": proposal,
    }
    if diagnostics:
        map_variables.sampling_metadata["production_proposal_stats"] = (
            variables.proposal_stats.copy()
        )
        map_variables.sampling_metadata["warmup_proposal_stats"] = warmup_stats
        map_variables.warmup_trace = warmup_trace
        map_variables.sampling_metadata["proposal_stat_names"] = np.array(
            [
                "inducing_support_rejections",
                "observation_support_rejections",
                "metropolis_rejections",
                "accepted_squared_displacement",
                "proposal_standard_deviation_sum",
            ]
        )

    return_arrays_to_host(map_variables, backend)
    return map_variables, dVect, pVect


def run_warmup(
    variables, data, burnIter, stabilizeIter, burn_in, adaptation, diagnostics,
):
    """Run annealing, adaptation and fixed-width discarded equilibration."""
    warmup_trace = []
    # Redefine perturbation magnitude for samples
    variables.epsilon = np.ones(variables.nProposals)

    # Initial temp and cooling rate for targeted annealing
    initTemp = 7.5
    coolRate = np.log(initTemp) / max(1, burnIter)
    log_width = np.zeros(variables.nProposals)
    window_acceptance = np.zeros(variables.nProposals)

    def adapt_width(accepted, iteration, total, target, phase):
        nonlocal window_acceptance, log_width
        window_acceptance += accepted
        if (iteration + 1) % 50 == 0 or iteration + 1 == total:
            count = (iteration % 50) + 1
            rate = window_acceptance / count
            gain = 2.0 * (1 + iteration // 50) ** -0.6
            log_width += gain * (rate - target)
            variables.epsilon = np.exp(log_width)
            if not np.all(np.isfinite(variables.epsilon)) or np.any(
                variables.epsilon <= 0
            ):
                raise FloatingPointError(
                    "Warmup proposal widths became invalid"
                )
            window_acceptance = np.zeros(variables.nProposals)
            if diagnostics:
                warmup_trace.append(
                    (
                        phase,
                        iteration + 1,
                        variables.epsilon.copy(),
                        rate.copy(),
                        variables.proposal_stats.copy(),
                    )
                )

    ## BURN IN 1 ##
    # Burn in super aggressively but do not save the samples,
    # updating proposal size to maintain healthy acceptance rate
    accVect = np.zeros(variables.nProposals)
    for i in range(burnIter):
        variables.temperature = functions.expCooling(i, initTemp, coolRate)
        print(f"Annealing Iteration {i + 1}/{burnIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        accVectNorm = 100 * accVect / (i + 1)
        if adaptation == "diminishing":
            adapt_width(accCount, i, burnIter, 0.3, "annealing")
        else:
            variables.epsilon = np.where(
                accVectNorm > 40,
                variables.epsilon * 1.5,
                np.where(
                    accVectNorm < 20,
                    variables.epsilon * 0.5,
                    variables.epsilon,
                ),
            )

        print(f"({time.time() - t:.3f}s)")

    ## Stabilize proposal Magnitude ##
    # Set temperature back to 1 and acceptance to 0, and iterate
    # to tune proposal magnitudes at temperature one.
    variables.temperature = 1
    window_acceptance = np.zeros(variables.nProposals)
    accVect = np.zeros(variables.nProposals)
    for i in range(stabilizeIter):
        print(f"Stabilization Iteration {i + 1}/{stabilizeIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        if adaptation == "diminishing":
            adapt_width(accCount, i, stabilizeIter, 0.25, "adaptation")
        elif i > 10:
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

    return warmup_trace


def run_production(variables, data, nIter, backend):
    """Retain every production draw and collect the highest-posterior state."""
    production_acceptance = np.zeros(variables.nProposals)
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
        production_acceptance += accCount

        # Save the sample
        dVect[i] = (
            variables.dIndu.get() if backend == "gpu" else variables.dIndu
        )
        pVect[i] = variables.P
        if variables.P >= map_variables.P:
            map_variables = copy.deepcopy(variables)

        print(f"({time.time() - t:.3f}s)")

    return map_variables, dVect, pVect, production_acceptance


def return_arrays_to_host(map_variables, backend):
    """Synchronize CUDA and return arrays in physical coordinates."""
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

