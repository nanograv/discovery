"""
Pytest tests for makegp_ecorr bug fixes (PR #100, Issue #99)
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data"


def test_single_backend_pulsar(data_dir):
    """Test single-backend bin bug fix"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "single_backend_pulsar.feather")

    # Should not raise an error
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    # Should have non-empty F matrix
    assert hasattr(gp_ecorr, 'F')
    assert gp_ecorr.F.shape[0] == len(psr.toas)


def test_empty_epoch_pulsar(data_dir):
    """Test empty-epoch bug fix"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "empty_epoch_pulsar.feather")

    # Should not raise an error even with backend having no simultaneous TOAs
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    assert hasattr(gp_ecorr, 'F')


def test_multi_backend_pulsar(data_dir):
    """Test multi-backend control case"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "multi_backend_pulsar.feather")

    # Should work correctly
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    assert hasattr(gp_ecorr, 'F')
    assert gp_ecorr.F.shape[0] == len(psr.toas)


def test_missing_bin_zero_bug(data_dir):
    """Test that makegp_ecorr includes bin 0 when mask has no zeros.
    Fixes issue #99.
    """
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "single_backend_pulsar.feather")

    # For a single-backend pulsar, the mask should have no zeros
    backend_flags = ds.selection_backend_flags(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]

    # Verify this is a single-backend case where mask has no zeros
    assert len(backends) == 1, "This test requires a single-backend pulsar"
    mask = masks[0]
    assert np.all(mask), "Mask should have no zeros for single-backend pulsar"

    # Check what bins quantize returns
    masked_toas = psr.toas * mask
    bins = ds.quantize(masked_toas)
    unique_bins = np.unique(bins)
    bins_max = bins.max()
    print(f"Unique bins from quantize: {unique_bins}")
    print(f"bins.max(): {bins_max}")

    # Expected behavior: if bins go from 0 to bins_max, we should have bins_max + 1 columns
    # Buggy behavior: range(1, bins.max() + 1) gives only bins_max columns (missing bin 0)
    if 0 in unique_bins and bins_max > 0:
        # If bin 0 exists and there are other bins, we should have bins_max + 1 columns
        expected_num_columns = bins_max + 1
        # But the buggy code using range(1, bins.max() + 1) would only give bins_max columns
        buggy_num_columns = bins_max
    else:
        # If bin 0 doesn't exist or there's only one bin, the current code might be correct
        expected_num_columns = len(unique_bins)
        buggy_num_columns = expected_num_columns

    # Call makegp_ecorr
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    # Verify the F matrix has the correct number of columns
    assert hasattr(gp_ecorr, 'F'), "GP should have F attribute"
    print(f"gp_ecorr.F.shape: {gp_ecorr.F.shape}")
    print(f"Expected number of columns (including bin 0): {expected_num_columns}")
    print(f"Buggy number of columns (missing bin 0): {buggy_num_columns}")
    print(f"Actual number of columns: {gp_ecorr.F.shape[1]}")

    # This assertion will fail if bin 0 is being skipped
    # If 0 is in unique_bins and bins_max > 0, we should have bins_max + 1 columns
    if 0 in unique_bins and bins_max > 0:
        assert gp_ecorr.F.shape[1] == expected_num_columns, (
            f"F matrix should have {expected_num_columns} columns (bins 0 through {bins_max}), "
            f"but got {gp_ecorr.F.shape[1]}. This indicates bin 0 is being skipped by "
            f"range(1, bins.max() + 1)."
        )