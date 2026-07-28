"""Tests for the working_dtype config knob.

Exercises utils.config(working=...) and the derived utils.working_dtype() /
utils.single_precision / utils.to_working() accessors.

Each test calls config() explicitly so the module is left in its default
state (jax, float64) by the teardown fixture.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import discovery.utils as utils


@pytest.fixture(autouse=True)
def restore_default():
    """Restore the default jax+float64 config after every test."""
    yield
    jax.config.update("jax_enable_x64", True)
    utils.config(backend='jax', factor='cholesky')


# A1 — default working_dtype is float64; single_precision is False
def test_a1_default_is_float64():
    utils.config(backend='jax', factor='cholesky')
    assert utils.working_dtype() == jnp.float64
    assert utils.single_precision is False


# A2 — selecting float32 with x64 on → working_dtype float32, single_precision True
def test_a2_float32_with_x64():
    jax.config.update("jax_enable_x64", True)
    utils.config(backend='jax', factor='cholesky', working=jnp.float32)
    assert utils.working_dtype() == jnp.float32
    assert utils.single_precision is True


# A3 — selecting float32 with x64 off → raises
def test_a3_float32_without_x64_raises():
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(AssertionError, match="x64 enabled"):
        utils.config(backend='jax', factor='cholesky', working=jnp.float32)


# A4 — numpy backend → working_dtype is float64, single_precision False
def test_a4_numpy_backend():
    utils.config(backend='numpy', factor='cholesky')
    assert utils.working_dtype() == np.float64
    assert utils.single_precision is False


# to_working casts to the current working dtype
def test_to_working_float64():
    utils.config(backend='jax', factor='cholesky')
    a = np.array([1.0, 2.0], dtype=np.float32)
    out = utils.to_working(a)
    assert out.dtype == jnp.float64


def test_to_working_float32():
    jax.config.update("jax_enable_x64", True)
    utils.config(backend='jax', factor='cholesky', working=jnp.float32)
    a = np.array([1.0, 2.0], dtype=np.float64)
    out = utils.to_working(a)
    assert out.dtype == jnp.float32
