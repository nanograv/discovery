"""Fixtures for matrix.py-vs-metamath parity tests."""

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402


DATA = Path(__file__).resolve().parents[2] / "data"
B1855 = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"


@pytest.fixture(scope="session")
def psr():
    """Single pulsar fixture — B1855+09."""
    return ds.Pulsar.read_feather(B1855)


@pytest.fixture(scope="session")
def psrs():
    """Multi-pulsar fixture — 3 pulsars for Global/Array tests."""
    files = [
        DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather",
        DATA / "v1p1_de440_pint_bipm2019-J0023+0923.feather",
        DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather",
    ]
    return [ds.Pulsar.read_feather(f) for f in files]
