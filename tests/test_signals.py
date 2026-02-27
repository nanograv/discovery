"""Tests for make_combined_crn signature merging and numerical correctness."""

import inspect
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery.signals import make_combined_crn


# A PSD with non-overlapping parameter names, for testing the no-rename path.
def _alt_psd(f, df, alpha, log10_ref):
    return (10.0 ** (2.0 * log10_ref)) * f ** (-alpha) * df


def _make_freqs(n_total=30, tspan_years=20):
    """Return (f, df) arrays with sin/cos pairs (2*n_total elements)."""
    tspan = tspan_years * 365.25 * 86400
    f = jnp.repeat(jnp.arange(1, n_total + 1) / tspan, 2)
    df = jnp.ones_like(f) / tspan
    return f, df


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

class TestMakeCombinedCrnSignature:

    def test_same_function_default_prefix(self):
        """Overlapping params get crn_ prefix when same function is passed twice."""
        combined = make_combined_crn(14, ds.powerlaw, ds.powerlaw)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'crn_log10_A', 'crn_gamma'], \
            f"Got args: {args}"

    def test_same_function_no_prefix_ties_params(self):
        """crn_prefix=None with same function: params are tied, no duplication."""
        combined = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix=None)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma'], f"Got args: {args}"

    def test_no_overlap_no_rename(self):
        """Non-overlapping param names require no renaming."""
        combined = make_combined_crn(14, ds.powerlaw, _alt_psd)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'alpha', 'log10_ref'], \
            f"Got args: {args}"

    def test_custom_prefix(self):
        """Custom prefix is applied to overlapping CRN param names."""
        combined = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix='gw_')
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'gw_log10_A', 'gw_gamma'], \
            f"Got args: {args}"


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

class TestMakeCombinedCrnValues:

    def test_same_function_separate_params(self):
        """phi = irn(A1,g1) + crn(A2,g2) on CRN bins; irn(A1,g1) elsewhere."""
        n_crn = 14
        combined = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        crn_log10_A, crn_gamma = -15.0, 13 / 3

        phi = combined(f, df, log10_A, gamma, crn_log10_A, crn_gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)
        crn = ds.powerlaw(f[:2 * n_crn], df[:2 * n_crn], crn_log10_A, crn_gamma)

        np.testing.assert_allclose(phi[:2 * n_crn], irn[:2 * n_crn] + crn, rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_same_function_tied_params(self):
        """crn_prefix=None + same function: CRN bins = 2 * irn; rest unchanged."""
        n_crn = 14
        combined = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw, crn_prefix=None)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        phi = combined(f, df, log10_A, gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)

        # Both PSDs receive identical params -> CRN contribution doubles the IRN value
        np.testing.assert_allclose(phi[:2 * n_crn], 2.0 * irn[:2 * n_crn], rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_no_overlap_values(self):
        """Non-overlapping PSDs: CRN bins = irn + alt_psd; rest = irn only."""
        n_crn = 14
        combined = make_combined_crn(n_crn, ds.powerlaw, _alt_psd)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        alpha, log10_ref = 3.0, -14.0

        phi = combined(f, df, log10_A, gamma, alpha, log10_ref)
        irn = ds.powerlaw(f, df, log10_A, gamma)
        crn = _alt_psd(f[:2 * n_crn], df[:2 * n_crn], alpha, log10_ref)

        np.testing.assert_allclose(phi[:2 * n_crn], irn[:2 * n_crn] + crn, rtol=1e-6)
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)

    def test_n_crn_boundary(self):
        """CRN only affects exactly the first 2*n_crn bins."""
        n_crn = 5
        combined = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        crn_log10_A, crn_gamma = -15.0, 13 / 3

        phi = combined(f, df, log10_A, gamma, crn_log10_A, crn_gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)

        # Bins beyond n_crn are untouched
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)
        # Bins within n_crn are strictly larger than IRN alone
        assert np.all(phi[:2 * n_crn] > irn[:2 * n_crn])
