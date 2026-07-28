"""Pivot-amplitude power-law parameterization.

The pivot reparameterization samples the amplitude at a sensitivity-weighted
pivot frequency instead of 1/yr, decorrelating amplitude and slope. It must
produce an identical spectrum, be affine with unit Jacobian, and use an
unambiguous public parameter name.
"""

import inspect
import math

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from discovery import const  # noqa: E402
from discovery import signals as ds_sig  # noqa: E402


def test_pivot_amplitude_spectrum_equals_reference_amplitude_spectrum():
    """A pivot amplitude, converted to the reference amplitude, gives exactly
    the same PSD as the standard power law."""
    f_pivot = 3.0e-9
    param = ds_sig.PowerLawParameterization(slope_pivot_frequency=f_pivot)
    pl_pivot = ds_sig.make_powerlaw_pivot(f_pivot=f_pivot, parameterization=param)
    pl_ref = ds_sig.make_powerlaw()

    f = np.array([1.0e-9, 3.0e-9, 1.0e-8, 3.0e-8])
    df = np.full_like(f, 1.0e-9)
    log10_A_pivot = -14.3
    gamma = 3.7

    log10_A_ref = ds_sig.reference_log10_amplitude(
        log10_A_pivot, gamma, f_pivot=f_pivot, parameterization=param)
    got = np.asarray(pl_pivot(f, df, log10_A_pivot, gamma))
    expected = np.asarray(pl_ref(f, df, log10_A_ref, gamma))
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_pivot_transform_has_unit_jacobian():
    """The map (log10_A_pivot, gamma) -> (log10_A_ref, gamma) is affine with unit
    determinant, so no density Jacobian is needed."""
    f_pivot = 5.0e-9
    param = ds_sig.PowerLawParameterization(slope_pivot_frequency=f_pivot)

    def transform(coords):
        a_pivot, gamma = coords[0], coords[1]
        a_ref = ds_sig.reference_log10_amplitude(
            a_pivot, gamma, f_pivot=f_pivot, parameterization=param)
        return jnp.stack([a_ref, gamma])

    jac = np.asarray(jax.jacfwd(transform)(jnp.array([-14.0, 4.0])))
    assert np.isclose(np.linalg.det(jac), 1.0, atol=1e-12)
    # Explicitly upper-triangular with unit diagonal.
    shift = 0.5 * math.log10(f_pivot / const.fyr)
    assert np.allclose(jac, np.array([[1.0, shift], [0.0, 1.0]]), atol=1e-12)


def test_sensitivity_weighted_pivot_matches_definition():
    """log(f_pivot) is the sensitivity-weighted mean of log(f_j), with weights
    tr(F_j^T N0^-1 F_j)."""
    freqs = np.array([1.0e-9, 2.0e-9, 4.0e-9])
    weights = np.array([3.0, 1.0, 0.5])
    got = ds_sig.sensitivity_weighted_pivot_frequency(freqs, weights)
    expected = float(np.exp(np.sum(weights * np.log(freqs)) / np.sum(weights)))
    assert np.isclose(got, expected)

    # fourier_sensitivity_weights sums the sine+cosine trace per frequency.
    n_toa, n_comp = 6, 3
    rng = np.random.default_rng(0)
    fmat = rng.standard_normal((n_toa, 2 * n_comp))
    n0 = np.linspace(1.0, 2.0, n_toa)

    class _Ref:
        def solve(self, rhs):
            rhs = np.asarray(rhs)
            d = n0[:, None] if rhs.ndim == 2 else n0
            return rhs / d, float(np.sum(np.log(n0)))

    w = ds_sig.fourier_sensitivity_weights(fmat, _Ref())
    expected_w = np.array([
        fmat[:, 2 * j] @ (fmat[:, 2 * j] / n0)
        + fmat[:, 2 * j + 1] @ (fmat[:, 2 * j + 1] / n0)
        for j in range(n_comp)
    ])
    assert np.allclose(w, expected_w)


def test_pivot_parameter_names_are_unambiguous():
    """The pivoted PSD's amplitude argument is log10_A_pivot (never a reused
    log10_A), and the reference amplitude is a separate decoded quantity."""
    pl_pivot = ds_sig.make_powerlaw_pivot(f_pivot=3.0e-9)
    args = inspect.signature(pl_pivot).parameters
    assert "log10_A_pivot" in args
    assert "log10_A" not in args  # no ambiguous reuse of the reference name

    # Fixed-gamma variant keeps the pivot name and drops the gamma argument.
    pl_fixed = ds_sig.make_powerlaw_pivot(f_pivot=3.0e-9, gamma=4.33)
    fixed_args = inspect.signature(pl_fixed).parameters
    assert "log10_A_pivot" in fixed_args
    assert "gamma" not in fixed_args

    # The decoded reference amplitude is a distinct value at 1/yr.
    ref = ds_sig.reference_log10_amplitude(-14.0, 4.33, f_pivot=3.0e-9)
    assert ref != -14.0
