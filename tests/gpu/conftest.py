"""Actual CUDA setup; scientific tests are added only in Stage 3."""

import os

import pytest


@pytest.fixture(scope="module")
def cupy_gpu():
    """Require real CUDA execution, honoring the required-hardware policy."""
    try:
        import cupy as cp

        device = cp.cuda.Device(0)
        device.use()
        result = cp.sum(cp.arange(32, dtype=cp.float64) ** 2)
        device.synchronize()
        value = float(result.get())
    except Exception as error:
        message = f"CUDA execution unavailable: {error}"
        if os.environ.get("GPD_REQUIRE_GPU") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    assert value == 10416.0
    yield cp
    device.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
