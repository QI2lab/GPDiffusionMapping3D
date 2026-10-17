"""Physical diffusion recovery through the CPU file-input pipeline."""

import pytest

from scripts.file_physics import CASES, validate_file_physics

pytestmark = [pytest.mark.cpu, pytest.mark.integration]


@pytest.mark.parametrize("case", CASES)
def test_serialized_linked_tracks_recover_diffusion(tmp_path, case):
    validate_file_physics(tmp_path, "cpu", case)
