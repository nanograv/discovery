"""`utils.make_uind` — the single definition consumed by both kernel routes.

`matrix.py:35` imports `make_uind` from `utils`, so the metamath and matrix
routes share one implementation and one fix.
"""
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
from discovery import utils  # noqa: E402
from discovery import matrix  # noqa: E402


@pytest.fixture
def data_dir():
    """`tests/data` — the small hand-built pulsars, not the repo-root release
    data that `conftest.DATA` points at."""
    return Path(__file__).resolve().parents[1] / "data"


def test_single_definition_is_shared_by_both_routes():
    """matrix.py imports the utils definition; there is nothing else to fix."""
    assert matrix.make_uind is utils.make_uind


def test_variable_epoch_sizes_pad_with_zero():
    # 5 TOAs, 2 epochs. Epoch 0 holds TOAs {0, 1, 2, 4}; epoch 1 holds {2, 3}.
    U = np.array([[1, 0],
                  [1, 0],
                  [1, 1],
                  [0, 1],
                  [1, 0]], dtype=float)

    Uind = utils.make_uind(U)

    # (n_epoch, max_per_epoch + 1), TOA indices shifted by +1, 0 as sentinel.
    assert Uind.shape == (2, 5)
    assert Uind.dtype == np.dtype('i')
    np.testing.assert_array_equal(Uind[0], [1, 2, 3, 5, 0])
    np.testing.assert_array_equal(Uind[1], [3, 4, 0, 0, 0])


def test_empty_basis_returns_empty_index_table():
    """An ECORR selection matching no TOAs gives U with zero columns; the old
    body called `jnp.max` on an empty array and crashed."""
    Uind = utils.make_uind(np.zeros((5, 0)))

    assert Uind.shape == (0, 1)
    assert Uind.dtype == np.dtype('i')


@pytest.mark.parametrize("kernels", ["matrix", "metamath"])
def test_empty_ecorr_selection_builds_and_evaluates(data_dir, kernels):
    """End-to-end under each kernel mode: a pulsar with a backend that has no
    simultaneous TOAs builds an ECORR GP and evaluates a finite likelihood."""
    ds.config(kernels=kernels)
    try:
        psr = ds.Pulsar.read_feather(data_dir / "empty_epoch_pulsar.feather")

        model = ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
        ])

        logl = model.logL(ds.sample_uniform(model.logL.params))
    finally:
        ds.config(kernels="matrix")

    assert np.isfinite(float(logl))
