"""Physical diffusion recovery through the CPU file-input pipeline."""

import pytest

from tests.support.file_physics import CASES, validate_file_physics

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", CASES)
def test_serialized_linked_tracks_recover_diffusion(
    tmp_path, case, serialized_track_sources
):
    validate_file_physics(
        tmp_path, "cpu", case,
        source=serialized_track_sources[case == "legacy-csv"],
    )
