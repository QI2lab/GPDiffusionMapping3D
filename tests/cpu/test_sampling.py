"""Actual CPU posterior sampling against independently integrated densities."""

import pytest
from scripts.posterior_oracle import validate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("proposal", ["additive", "blocked"])
def test_small_positive_diffusion_posterior(
    dimensions, proposal, posterior_problems, posterior_references
):
    validate(
        "cpu", dimensions, proposal=proposal,
        prepared_problem=posterior_problems[dimensions],
        reference_values=posterior_references[dimensions],
    )
