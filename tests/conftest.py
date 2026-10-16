# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Reusable deterministic inputs; mutable sampler states stay local to tests."""

import contextlib
import io

import numpy as np
import pytest

from gpdiffusionmapping import functions as f
from gpdiffusionmapping import model
from tests.support.posterior_oracle import problem, reference
from tests.support.validate_2d import parameters, simulate as simulate_2d
from tests.support.validate_3d import prepared, simulate as simulate_3d


@pytest.fixture(scope="session")
def gp_states():
    """Share read-only GP geometry for the mathematical identities."""
    data_3d = prepared(count=300)
    states = {}
    for name, data, config in (
        ("2d", simulate_2d(100), parameters()),
        *(
            (f"3d-{size}", data_3d, {
                **model.PARAMETERS,
                "nInduX": size, "nInduY": size, "nInduZ": size,
                "nFineX": 3, "nFineY": 3, "nFineZ": 3,
                "covLambda": 0.5, "covL": 1.5,
            })
            for size in (3, 5)
        ),
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            state = f.initialization(config, data, backend="cpu")
        for value in vars(state).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        for value in vars(data).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        states[name] = data, state
    return states


@pytest.fixture(scope="session")
def posterior_problems():
    """Reuse deterministic one- and two-inducing-value oracle inputs."""
    problems = {dimensions: problem(dimensions) for dimensions in (1, 2)}
    for data, _, covariance, weights, _ in problems.values():
        for value in (*vars(data).values(), covariance, weights):
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    return problems


@pytest.fixture(scope="session")
def posterior_references(posterior_problems):
    """Integrate each independent density once at both quadrature orders."""
    references = {}
    for dimensions, prepared_problem in posterior_problems.items():
        expected, thresholds = reference(
            dimensions, prepared_problem=prepared_problem
        )
        refined, _ = reference(
            dimensions, order=256, thresholds=thresholds,
            prepared_problem=prepared_problem,
        )
        for value in (expected, refined, thresholds):
            value.flags.writeable = False
        references[dimensions] = expected, refined, thresholds
    return references


@pytest.fixture(scope="session")
def serialized_track_sources():
    """Share physical source tracks, keeping file inference independent."""
    sources = {
        legacy: simulate_3d(seed=44, irregular=not legacy)
        for legacy in (False, True)
    }
    for source in sources.values():
        for value in source:
            value.flags.writeable = False
    return sources
