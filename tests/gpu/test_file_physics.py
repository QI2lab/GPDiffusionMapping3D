"""Physical diffusion recovery through CSV/Parquet on actual CUDA hardware."""

import pytest

from scripts.file_physics import CASES, validate_file_physics

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("case", CASES)
def test_serialized_linked_tracks_recover_diffusion(cupy_gpu, tmp_path, case):
    validate_file_physics(tmp_path, "gpu", case)
