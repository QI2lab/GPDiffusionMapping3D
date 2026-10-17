# SPDX-License-Identifier: MIT
# Project attribution: see LICENSE and NOTICE.

"""Independent per-cell diffusion recovery through automatic CPU batching."""

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
