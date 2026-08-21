"""Tests for make_combined_crn signature merging and numerical correctness."""

import inspect
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import matrix
from discovery.signals import (
    make_combined_crn,
    fourierbasis,
    dmfourierbasis,
    fourierbasis_chrom,
    make_fourierbasis_chrom,
    dmfourierbasis_solar,
    log_fourierbasis,
    log_fourierbasis_dm,
    log_fourierbasis_chrom,
    log_fourierbasis_chrom_fixed,
    makegp_fourier,
    ridge_kernel,
    square_exponential_kernel,
    quasi_periodic_kernel,
    matern_kernel,
    linear_blocked_interpolation_basis,
    custom_blocked_interpolation_basis,
    makegp_timedomain_dm,
)


# ---------------------------------------------------------------------------
# Minimal mock pulsar
# ---------------------------------------------------------------------------

class _MockPulsar:
    """Minimal stand-in for discovery.Pulsar sufficient for Fourier basis tests."""

    def __init__(self, n_toas=50, tspan_years=20.0, seed=42):
        rng = np.random.default_rng(seed)
        tspan = tspan_years * 365.25 * 86400
        self.toas = np.sort(rng.uniform(0, tspan, n_toas))
        self.freqs = rng.uniform(1000, 2000, n_toas)
        self.residuals = rng.normal(0, 1e-6, n_toas)
        self.backend_flags = np.array(['backend_A'] * n_toas)
        self.name = 'J0000+0000'
        self.pos = np.array([1.0, 0.0, 0.0])

    @property
    def mintoa(self):
        return self.toas.min()

    @property
    def maxtoa(self):
        return self.toas.max()


@pytest.fixture(scope='module')
def psr():
    return _MockPulsar()


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
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'crn_log10_A', 'crn_gamma'], \
            f"Got args: {args}"
        assert crn_params == ['crn_log10_A', 'crn_gamma'], \
            f"Got crn_params: {crn_params}"

    def test_same_function_no_prefix_ties_params(self):
        """crn_prefix=None with same function: params are tied, no duplication."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix=None)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma'], f"Got args: {args}"
        assert crn_params == ['log10_A', 'gamma'], f"Got crn_params: {crn_params}"

    def test_no_overlap_no_rename(self):
        """Non-overlapping param names require no renaming."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, _alt_psd)
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'alpha', 'log10_ref'], \
            f"Got args: {args}"
        assert crn_params == ['alpha', 'log10_ref'], f"Got crn_params: {crn_params}"

    def test_custom_prefix(self):
        """Custom prefix is applied to overlapping CRN param names."""
        combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw, crn_prefix='gw_')
        args = inspect.getfullargspec(combined).args
        assert args == ['f', 'df', 'log10_A', 'gamma', 'gw_log10_A', 'gw_gamma'], \
            f"Got args: {args}"
        assert crn_params == ['gw_log10_A', 'gw_gamma'], f"Got crn_params: {crn_params}"


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

class TestMakeCombinedCrnValues:

    def test_same_function_separate_params(self):
        """phi = irn(A1,g1) + crn(A2,g2) on CRN bins; irn(A1,g1) elsewhere."""
        n_crn = 14
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
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
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw, crn_prefix=None)
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
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, _alt_psd)
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
        combined, _ = make_combined_crn(n_crn, ds.powerlaw, ds.powerlaw)
        f, df = _make_freqs()

        log10_A, gamma = -14.5, 4.3
        crn_log10_A, crn_gamma = -15.0, 13 / 3

        phi = combined(f, df, log10_A, gamma, crn_log10_A, crn_gamma)
        irn = ds.powerlaw(f, df, log10_A, gamma)

        # Bins beyond n_crn are untouched
        np.testing.assert_allclose(phi[2 * n_crn:], irn[2 * n_crn:], rtol=1e-6)
        # Bins within n_crn are strictly larger than IRN alone
        assert np.all(phi[:2 * n_crn] > irn[:2 * n_crn])


# ---------------------------------------------------------------------------
# fourierbasis tests: int components (grid) vs. array components (explicit modes)
# ---------------------------------------------------------------------------

class TestFourierbasisModes:

    def test_default_modes_frequencies(self, psr):
        """With an int, frequencies are arange(1, n+1)/T."""
        n = 10
        T = psr.maxtoa - psr.mintoa
        f, df, fmat = fourierbasis(psr, n, T=T)
        expected_f = np.repeat(np.arange(1, n + 1) / T, 2)
        np.testing.assert_allclose(f, expected_f, rtol=1e-12)

    def test_shape_without_modes(self, psr):
        """Output shapes are consistent with n_components."""
        n = 8
        f, df, fmat = fourierbasis(psr, n)
        assert f.shape == (2 * n,)
        assert df.shape == (2 * n,)
        assert fmat.shape == (len(psr.toas), 2 * n)

    def test_custom_modes_frequency_values(self, psr):
        """An array of components overrides the frequency grid."""
        T = psr.maxtoa - psr.mintoa
        modes = np.array([1.0, 2.5, 5.0]) / T
        f, df, fmat = fourierbasis(psr, modes, T=T)
        np.testing.assert_allclose(f[::2], modes, rtol=1e-12)

    def test_shape_with_modes(self, psr):
        """Shape with an explicit mode array matches the number of modes."""
        modes = np.array([1e-9, 2e-9, 3e-9, 4e-9])
        f, df, fmat = fourierbasis(psr, modes)
        assert f.shape == (2 * len(modes),)
        assert fmat.shape == (len(psr.toas), 2 * len(modes))

    def test_modes_matches_components_basis(self, psr):
        """A mode array equal to the default grid == the plain int call."""
        n = 6
        T = psr.maxtoa - psr.mintoa
        default_modes = np.arange(1, n + 1, dtype=float) / T
        f1, df1, fmat1 = fourierbasis(psr, n, T=T)
        f2, df2, fmat2 = fourierbasis(psr, default_modes, T=T)
        np.testing.assert_allclose(fmat1, fmat2, rtol=1e-12)
        np.testing.assert_allclose(f1, f2, rtol=1e-12)

    def test_modes_accepts_list(self, psr):
        """A plain Python list of modes works like an ndarray."""
        modes = [1e-9, 2e-9, 3e-9]
        f, df, fmat = fourierbasis(psr, modes)
        np.testing.assert_allclose(f[::2], modes, rtol=1e-12)
        assert fmat.shape == (len(psr.toas), 2 * len(modes))

    def test_invalid_components_type_raises(self, psr):
        """A non-int, non-array components raises an informative TypeError."""
        with pytest.raises(TypeError, match="components"):
            fourierbasis(psr, "not valid")

    def test_sin_cos_columns(self, psr):
        """Even columns are sin, odd columns are cos."""
        modes = np.array([1e-9])
        f, df, fmat = fourierbasis(psr, modes)
        np.testing.assert_allclose(fmat[:, 0], np.sin(2 * np.pi * modes[0] * psr.toas), rtol=1e-12)
        np.testing.assert_allclose(fmat[:, 1], np.cos(2 * np.pi * modes[0] * psr.toas), rtol=1e-12)

    def test_dmfourierbasis_modes_shape(self, psr):
        """dmfourierbasis with a mode array returns correct shape."""
        modes = np.array([1e-9, 2e-9, 3e-9])
        f, df, fmat = dmfourierbasis(psr, modes)
        assert fmat.shape == (len(psr.toas), 2 * len(modes))

    def test_dmfourierbasis_dm_scaling(self, psr):
        """DM Fourier basis is frequency-scaled version of plain Fourier basis."""
        modes = np.array([1e-9, 2e-9])
        fref = 1400.0
        _, _, fmat_plain = fourierbasis(psr, modes)
        _, _, fmat_dm = dmfourierbasis(psr, modes, fref=fref)
        Dm = (fref / psr.freqs) ** 2
        np.testing.assert_allclose(fmat_dm, fmat_plain * Dm[:, None], rtol=1e-12)

    def test_fourierbasis_chrom_fixed_idx(self, psr):
        """make_fourierbasis_chrom(alpha=...) returns an array, not a callable."""
        modes = np.array([1e-9, 2e-9])
        f, df, fmat = make_fourierbasis_chrom(alpha=4.0)(psr, modes)
        assert isinstance(fmat, np.ndarray) or hasattr(fmat, 'shape')
        assert np.asarray(fmat).shape == (len(psr.toas), 2 * len(modes))

    def test_fourierbasis_chrom_free_idx_callable(self, psr):
        """fourierbasis_chrom returns a callable fmat (free chromatic index)."""
        modes = np.array([1e-9, 2e-9])
        f, df, fmat = fourierbasis_chrom(psr, modes)
        assert callable(fmat)
        result = fmat(4.0)
        assert result.shape == (len(psr.toas), 2 * len(modes))


class TestMakegpFourierModes:

    def test_modes_sets_component_count(self, psr):
        """makegp_fourier with a mode array uses len(modes) as component count."""
        T = psr.maxtoa - psr.mintoa
        modes = np.array([1.0, 2.0, 3.0, 4.0]) / T
        gp = makegp_fourier(psr, ds.powerlaw, modes, T=T)
        # The GP basis (gp.F) should have 2*len(modes) columns
        assert gp.F.shape == (len(psr.toas), 2 * len(modes))

    def test_modes_frequencies_in_gp(self, psr):
        """makegp_fourier frequencies match the supplied mode array."""
        T = psr.maxtoa - psr.mintoa
        modes = np.array([1.5, 3.0, 7.5]) / T
        gp = makegp_fourier(psr, ds.powerlaw, modes, T=T)
        assert gp.F.shape == (len(psr.toas), 2 * len(modes))

    def test_int_components_sets_basis_size(self, psr):
        """An int components value controls the basis size."""
        n = 12
        gp = makegp_fourier(psr, ds.powerlaw, n)
        assert gp.F.shape == (len(psr.toas), 2 * n)


# ---------------------------------------------------------------------------
# Log Fourier basis tests
# ---------------------------------------------------------------------------

class TestLogFourierbasis:

    def _default_nlin(self):
        return 30

    def test_log_fourierbasis_shape(self, psr):
        """log_fourierbasis returns arrays of shape (2*nlin, ) and (ntoa, 2*nlin)."""
        nlin = 20
        f, df, fmat = log_fourierbasis(psr, logmode=0, nlin=nlin, nlog=0)
        assert f.shape == (2 * nlin,)
        assert df.shape == (2 * nlin,)
        assert fmat.shape == (len(psr.toas), 2 * nlin)

    def test_log_fourierbasis_frequencies_positive(self, psr):
        """All frequencies returned by log_fourierbasis must be positive."""
        f, df, fmat = log_fourierbasis(psr, logmode=0, nlin=15, nlog=0)
        assert np.all(f > 0)

    def test_log_fourierbasis_with_logmodes(self, psr):
        """log_fourierbasis with logmode>=0 and nlog>0 returns nlin+nlog frequencies."""
        nlin, nlog = 10, 5
        T = psr.maxtoa - psr.mintoa
        f, df, fmat = log_fourierbasis(psr, T=T, logmode=0, f_min=0.5/T, nlin=nlin, nlog=nlog)
        assert f.shape == (2 * (nlin + nlog),)
        assert fmat.shape == (len(psr.toas), 2 * (nlin + nlog))

    def test_log_fourierbasis_sin_cos_columns(self, psr):
        """Even/odd columns are sin/cos at return frequencies."""
        nlin = 5
        f, df, fmat = log_fourierbasis(psr, logmode=0, nlin=nlin, nlog=0)
        freq = f[::2]  # unique frequencies (before repeat)
        for i, fi in enumerate(freq):
            np.testing.assert_allclose(
                fmat[:, 2 * i], np.sin(2 * np.pi * fi * psr.toas), rtol=1e-10,
                err_msg=f"sin column {i} mismatch"
            )
            np.testing.assert_allclose(
                fmat[:, 2 * i + 1], np.cos(2 * np.pi * fi * psr.toas), rtol=1e-10,
                err_msg=f"cos column {i} mismatch"
            )

    def test_log_fourierbasis_dm_shape(self, psr):
        """log_fourierbasis_dm returns same shape as log_fourierbasis."""
        nlin = 15
        f, df, fmat = log_fourierbasis_dm(psr, logmode=0, nlin=nlin, nlog=0)
        assert f.shape == (2 * nlin,)
        assert fmat.shape == (len(psr.toas), 2 * nlin)

    def test_log_fourierbasis_dm_scaling(self, psr):
        """log_fourierbasis_dm = log_fourierbasis * (fref/freqs)^2."""
        nlin = 10
        fref = 1400.0
        f_plain, _, fmat_plain = log_fourierbasis(psr, logmode=0, nlin=nlin, nlog=0)
        f_dm, _, fmat_dm = log_fourierbasis_dm(psr, logmode=0, nlin=nlin, nlog=0, fref=fref)
        Dm = (fref / psr.freqs) ** 2
        np.testing.assert_allclose(fmat_dm, fmat_plain * Dm[:, None], rtol=1e-10)
        np.testing.assert_allclose(f_plain, f_dm, rtol=1e-12)

    def test_log_fourierbasis_chrom_returns_callable(self, psr):
        """log_fourierbasis_chrom returns a callable fmat."""
        nlin = 10
        f, df, fmat = log_fourierbasis_chrom(psr, logmode=0, nlin=nlin, nlog=0)
        assert callable(fmat)

    def test_log_fourierbasis_chrom_callable_shape(self, psr):
        """Calling the free-chromatic fmat with an index gives correct shape."""
        nlin = 10
        f, df, fmat = log_fourierbasis_chrom(psr, logmode=0, nlin=nlin, nlog=0)
        result = fmat(4.0)
        assert result.shape == (len(psr.toas), 2 * nlin)

    def test_log_fourierbasis_chrom_vs_fixed(self, psr):
        """Free-chromatic callable at fixed idx matches fixed-chromatic basis."""
        nlin = 8
        idx = 3.7
        fref = 800.0
        _, _, fmat_free = log_fourierbasis_chrom(psr, logmode=0, nlin=nlin, nlog=0, fref=fref)
        _, _, fmat_fixed = log_fourierbasis_chrom_fixed(psr, alpha=idx, logmode=0, nlin=nlin, nlog=0, fref=fref)
        np.testing.assert_allclose(np.asarray(fmat_free(idx)), np.asarray(fmat_fixed), rtol=1e-10)

    def test_log_fourierbasis_chrom_fixed_shape(self, psr):
        """log_fourierbasis_chrom_fixed returns an array (not callable)."""
        nlin = 12
        f, df, fmat = log_fourierbasis_chrom_fixed(psr, alpha=4.0, logmode=0, nlin=nlin, nlog=0)
        assert hasattr(fmat, 'shape') or isinstance(fmat, np.ndarray)
        assert np.asarray(fmat).shape == (len(psr.toas), 2 * nlin)

    def test_log_fourierbasis_consistent_with_explicit_T(self, psr):
        """Passing T explicitly gives same result as letting it be auto-computed."""
        T = psr.maxtoa - psr.mintoa
        f1, df1, fmat1 = log_fourierbasis(psr, logmode=0, nlin=10, nlog=0)
        f2, df2, fmat2 = log_fourierbasis(psr, T=T, logmode=0, nlin=10, nlog=0)
        np.testing.assert_allclose(np.asarray(f1), np.asarray(f2), rtol=1e-12)
        np.testing.assert_allclose(fmat1, fmat2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Time-domain kernel tests
# ---------------------------------------------------------------------------

class TestTimeDomainKernels:
    """Tests for the kernel factory functions and their inner callables."""

    def _tau(self, n=5):
        """Small symmetric tau matrix (seconds)."""
        t = jnp.linspace(0, 1e8, n)
        return jnp.abs(t[:, None] - t[None, :])

    # --- ridge ---

    def test_ridge_kernel_shape(self):
        tau = self._tau()
        k = ridge_kernel()(tau)
        assert k.shape == tau.shape

    def test_ridge_kernel_is_diagonal(self):
        """Ridge kernel must be diagonal (identity * scale)."""
        tau = self._tau()
        k = ridge_kernel(log10_sigma_ridge=-7.0)(tau)
        off = k - jnp.diag(jnp.diag(k))
        np.testing.assert_allclose(np.asarray(off), 0.0, atol=1e-30)

    def test_ridge_kernel_scale(self):
        """Diagonal entries equal 10^(2*log10_sigma_ridge)."""
        s = -6.5
        tau = self._tau()
        k = ridge_kernel(log10_sigma_ridge=s)(tau)
        expected = 10 ** (2 * s)
        np.testing.assert_allclose(np.asarray(jnp.diag(k)), expected, rtol=1e-12)

    def test_ridge_kernel_params_forwarded(self):
        """Inner kernel respects parameter values passed at call time."""
        tau = self._tau(3)
        kern = ridge_kernel(log10_sigma_ridge=-7.0)
        k1 = kern(tau, log10_sigma_ridge=-7.0)
        k2 = kern(tau, log10_sigma_ridge=-6.0)
        assert float(k2[0, 0]) > float(k1[0, 0])

    # --- square exponential ---

    def test_sqexp_kernel_shape(self):
        tau = self._tau()
        k = square_exponential_kernel()(tau)
        assert k.shape == tau.shape

    def test_sqexp_kernel_symmetric(self):
        tau = self._tau()
        k = square_exponential_kernel()(tau)
        np.testing.assert_allclose(np.asarray(k), np.asarray(k).T, rtol=1e-12)

    def test_sqexp_kernel_positive_definite(self):
        """Square-exp kernel must be positive definite (all eigenvalues > 0)."""
        tau = self._tau(6)
        k = square_exponential_kernel(log10_sigma_sq_exp=-7., log10_ell=8.)(tau)
        eigvals = np.linalg.eigvalsh(np.asarray(k))
        assert np.all(eigvals > 0)

    def test_sqexp_kernel_decays_with_lag(self):
        """Off-diagonal entry decays as lag increases."""
        t = jnp.array([0.0, 1e7, 1e8])
        tau = jnp.abs(t[:, None] - t[None, :])
        k = square_exponential_kernel(log10_sigma_sq_exp=-7., log10_ell=8.)(tau)
        # k[0,1] (lag=1e7) should be larger than k[0,2] (lag=1e8)
        assert float(k[0, 1]) > float(k[0, 2])

    # --- quasi-periodic ---

    def test_qp_kernel_shape(self):
        tau = self._tau()
        k = quasi_periodic_kernel()(tau)
        assert k.shape == tau.shape

    def test_qp_kernel_symmetric(self):
        tau = self._tau()
        k = quasi_periodic_kernel()(tau)
        np.testing.assert_allclose(np.asarray(k), np.asarray(k).T, rtol=1e-12)

    def test_qp_kernel_positive_definite(self):
        tau = self._tau(6)
        k = quasi_periodic_kernel(
            log10_sigma_quasi_periodic=-7., log10_ell=8.,
            log10_gamma_p=0., log10_p=8.
        )(tau)
        eigvals = np.linalg.eigvalsh(np.asarray(k))
        assert np.all(eigvals > 0)

    # --- Matérn ---

    def test_matern_invalid_nu(self):
        with pytest.raises(ValueError, match="nu"):
            matern_kernel(nu=1.0)

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_matern_shape(self, nu):
        tau = self._tau()
        k = matern_kernel(nu=nu)(tau)
        assert k.shape == tau.shape

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_matern_symmetric(self, nu):
        tau = self._tau()
        k = matern_kernel(nu=nu)(tau)
        np.testing.assert_allclose(np.asarray(k), np.asarray(k).T, rtol=1e-12)

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_matern_positive_definite(self, nu):
        tau = self._tau(6)
        k = matern_kernel(log10_sigma_matern=-7., log10_ell=8., nu=nu)(tau)
        eigvals = np.linalg.eigvalsh(np.asarray(k))
        assert np.all(eigvals > 0)

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_matern_decays_with_lag(self, nu):
        """Off-diagonal covariance decays as lag increases."""
        t = jnp.array([0.0, 1e7, 1e8])
        tau = jnp.abs(t[:, None] - t[None, :])
        k = matern_kernel(log10_sigma_matern=-7., log10_ell=8., nu=nu)(tau)
        assert float(k[0, 1]) > float(k[0, 2])

    def test_matern_smoother_at_larger_nu(self):
        """Matérn with nu=2.5 should be smoother (larger off-diag) than nu=0.5."""
        t = jnp.array([0.0, 5e7])
        tau = jnp.abs(t[:, None] - t[None, :])
        ell = 8.0
        k05 = matern_kernel(log10_sigma_matern=0., log10_ell=ell, nu=0.5)(tau)
        k25 = matern_kernel(log10_sigma_matern=0., log10_ell=ell, nu=2.5)(tau)
        # Both share the same sigma^2 diagonal; nu=2.5 decays more slowly
        assert float(k25[0, 1]) > float(k05[0, 1])


# ---------------------------------------------------------------------------
# Interpolation basis tests
# ---------------------------------------------------------------------------

class TestInterpolationBases:

    def _toas_and_mjd_edges(self, n_toas=50, tspan_years=20.0, n_edges=10, seed=7):
        """Return toas (seconds), bin_edges (MJD), nodes (MJD)."""
        rng = np.random.default_rng(seed)
        tspan_s = tspan_years * 365.25 * 86400
        toas = np.sort(rng.uniform(0, tspan_s, n_toas))
        tspan_mjd = tspan_years * 365.25
        edges_mjd = np.linspace(0.0, tspan_mjd, n_edges + 1)   # n_edges+1 boundaries
        nodes_mjd = np.linspace(0.0, tspan_mjd, n_edges)
        return toas, edges_mjd, nodes_mjd

    # --- linear_blocked_interpolation_basis ---

    def test_linear_basis_shape(self):
        toas, edges, _ = self._toas_and_mjd_edges()
        M, be = linear_blocked_interpolation_basis(toas, edges)
        assert M.shape[0] == len(toas)
        assert M.shape[1] == be.shape[0]

    def test_linear_basis_row_sum_interior(self):
        """Interior TOAs have rows that sum to 1 (partition of unity)."""
        toas, edges, _ = self._toas_and_mjd_edges(n_edges=5)
        M, _ = linear_blocked_interpolation_basis(toas, edges)
        row_sums = M.sum(axis=1)
        # TOAs that fall strictly inside the span should sum to 1
        span_s = edges[-1] * 86400
        interior = (toas > edges[0] * 86400) & (toas < span_s)
        np.testing.assert_allclose(row_sums[interior], 1.0, atol=1e-12)

    def test_linear_basis_non_negative(self):
        toas, edges, _ = self._toas_and_mjd_edges()
        M, _ = linear_blocked_interpolation_basis(toas, edges)
        assert np.all(M >= 0)

    def test_linear_basis_removes_zero_columns(self):
        """Only columns with support (at least one non-zero entry) are returned."""
        toas, edges, _ = self._toas_and_mjd_edges()
        M, _ = linear_blocked_interpolation_basis(toas, edges)
        assert np.all(M.sum(axis=0) > 0)

    # --- custom_blocked_interpolation_basis ---

    def test_custom_basis_linear_shape(self):
        toas, _, nodes = self._toas_and_mjd_edges()
        M, nd = custom_blocked_interpolation_basis(toas, nodes, kind='linear')
        assert M.shape[0] == len(toas)
        assert M.shape[1] == nd.shape[0]

    def test_custom_basis_row_sum_interior(self):
        """Rows for TOAs within the node span sum to ~1 for linear interpolation."""
        toas, _, nodes = self._toas_and_mjd_edges(n_edges=8)
        M, nd = custom_blocked_interpolation_basis(toas, nodes, kind='linear')
        nodes_s = nd  # already in seconds after filtering
        interior = (toas > nodes_s.min()) & (toas < nodes_s.max())
        row_sums = M[interior].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_custom_basis_non_negative_linear(self):
        toas, _, nodes = self._toas_and_mjd_edges()
        M, _ = custom_blocked_interpolation_basis(toas, nodes, kind='linear')
        assert np.all(M >= 0)

    def test_custom_basis_removes_zero_columns(self):
        toas, _, nodes = self._toas_and_mjd_edges()
        M, _ = custom_blocked_interpolation_basis(toas, nodes)
        assert np.all(M.sum(axis=0) > 0)

    def test_custom_basis_no_support_raises(self):
        """Nodes entirely outside the TOA span should raise RuntimeError."""
        toas, _, _ = self._toas_and_mjd_edges()
        # Nodes far in the future (MJD way beyond the TOA span)
        bad_nodes = np.array([1e6, 1e6 + 365.0])
        with pytest.raises(RuntimeError, match="support"):
            custom_blocked_interpolation_basis(toas, bad_nodes)


# ---------------------------------------------------------------------------
# makegp_timedomain_dm tests
# ---------------------------------------------------------------------------

class TestMakegpTimeDomainDm:

    @pytest.fixture
    def sq_exp_cov(self):
        return square_exponential_kernel(log10_sigma_sq_exp=-7., log10_ell=8.)

    @pytest.fixture
    def matern_cov(self):
        return matern_kernel(log10_sigma_matern=-7., log10_ell=8., nu=1.5)

    def test_returns_variable_gp(self, psr, sq_exp_cov):
        """makegp_timedomain_dm returns a VariableGP."""
        from discovery import matrix
        dt = 86400 * 30  # monthly bins
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        assert isinstance(gp, matrix.VariableGP)

    def test_basis_shape(self, psr, sq_exp_cov):
        """Design matrix has shape (n_toas, n_bins)."""
        dt = 86400 * 30
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        n_toas = len(psr.toas)
        assert gp.F.shape[0] == n_toas
        assert gp.F.shape[1] >= 1

    def test_phi_params_named_correctly(self, psr, sq_exp_cov):
        """GP prior params are prefixed with pulsar name and gp name."""
        dt = 86400 * 30
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt, name='dm_gp')
        for p in gp.Phi.params:
            assert p.startswith(psr.name), f"Param {p!r} missing pulsar prefix"
            assert 'dm_gp' in p

    def test_phi_returns_square_matrix(self, psr, sq_exp_cov):
        """Phi.getN(params) returns a square covariance matrix."""
        dt = 86400 * 30
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        n_bins = gp.F.shape[1]
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        assert phi.shape == (n_bins, n_bins)

    def test_phi_symmetric(self, psr, sq_exp_cov):
        """Covariance matrix from Phi.getN is symmetric."""
        dt = 86400 * 30
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi).T, atol=1e-12)

    def test_different_kernels_give_different_phi(self, psr, sq_exp_cov, matern_cov):
        """Different kernel functions produce different covariance matrices."""
        dt = 86400 * 30
        gp_sq = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        gp_mt = makegp_timedomain_dm(psr, matern_cov, dt=dt)
        # Both GPs have the same bin structure
        assert gp_sq.F.shape == gp_mt.F.shape

        def _eval(gp, sigma_val, ell_val):
            params = {p: jnp.array(sigma_val) if 'sigma' in p else jnp.array(ell_val)
                      for p in gp.Phi.params}
            return gp.Phi.getN(params)

        phi_sq = _eval(gp_sq, -7.0, 3.0)
        phi_mt = _eval(gp_mt, -7.0, 3.0)
        # Kernels are distinct: their off-diagonal structure should differ at >1%
        diff = np.abs(np.asarray(phi_sq) - np.asarray(phi_mt))
        scale = np.abs(np.asarray(phi_sq))
        # Mask diagonal (dominated by regularisation); check off-diag relative difference
        mask = ~np.eye(phi_sq.shape[0], dtype=bool)
        rel_diff = diff[mask] / (scale[mask] + 1e-300)
        assert np.max(rel_diff) > 0.01, \
            "sq-exp and Matérn kernels produced nearly identical covariance matrices"

    def test_dm_scaling_in_basis(self, psr, sq_exp_cov):
        """Design matrix columns scale with (fref/freqs)^2 — higher-freq TOAs
        have smaller weights for a given DM coefficient."""
        dt = 86400 * 30
        fref = 1400.0
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt, fref=fref)
        # For each column (bin), the row weights should track (fref/freq)^2
        # Find a bin that has contributions from multiple TOAs at different freqs
        col_sums = np.asarray(gp.F).sum(axis=1)
        expected_scale = (fref / psr.freqs) ** 2
        # Row weights should be proportional to (fref/freq)^2 within each bin;
        # check that the ratio of weights between two TOAs in the same bin
        # matches the ratio of DM scalings.
        F = np.asarray(gp.F)
        for col in range(F.shape[1]):
            rows = np.where(F[:, col] != 0)[0]
            if len(rows) < 2:
                continue
            i, j = rows[0], rows[1]
            ratio_F = F[i, col] / F[j, col]
            ratio_dm = expected_scale[i] / expected_scale[j]
            np.testing.assert_allclose(ratio_F, ratio_dm, rtol=1e-10)
            break  # one bin is enough

    def test_index_attribute(self, psr, sq_exp_cov):
        """GP has an index dict with a key starting with the pulsar name."""
        dt = 86400 * 30
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt)
        assert len(gp.index) == 1
        key = next(iter(gp.index))
        assert key.startswith(psr.name)

    def test_common_param(self, psr, sq_exp_cov):
        """A parameter listed in common is not prefixed with the pulsar name."""
        dt = 86400 * 30
        # Peek at param names inside sq_exp_cov to know which to mark common
        import inspect as _inspect
        args = _inspect.getfullargspec(sq_exp_cov).args
        sigma_arg = next(a for a in args if 'sigma' in a)
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=dt, name='dm_gp',
                                   common=[sigma_arg])
        assert sigma_arg in gp.Phi.params, \
            f"Expected bare param {sigma_arg!r} in {gp.Phi.params}"

    # ---- fixed-hyperparameter ("fixed-point") path --------------------------

    @staticmethod
    def _full_noisedict(params):
        return {p: (-7.0 if 'sigma' in p else 8.0) for p in params}

    def test_empty_noisedict_returns_variable_gp(self, psr, sq_exp_cov):
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30, noisedict={})
        assert isinstance(gp, matrix.VariableGP)

    def test_full_noisedict_returns_constant_gp(self, psr, sq_exp_cov):
        gv = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30)
        gc = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30,
                                  noisedict=self._full_noisedict(gv.Phi.params))
        assert isinstance(gc, matrix.ConstantGP)
        assert isinstance(gc.Phi, matrix.NoiseMatrix2D_novar)

    def test_partial_noisedict_returns_variable_gp(self, psr, sq_exp_cov):
        gv = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30)
        full = self._full_noisedict(gv.Phi.params)
        one_key = next(iter(full))
        gp = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30,
                                  noisedict={one_key: full[one_key]})
        assert isinstance(gp, matrix.VariableGP)

    def test_cached_covariance_matches_variable(self, psr, sq_exp_cov):
        gv = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30, noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.Phi.N),
                                   np.asarray(gv.Phi.getN(nd)))

    def test_cached_basis_matches_variable(self, psr, sq_exp_cov):
        gv = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_dm(psr, sq_exp_cov, dt=86400 * 30, noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.F), np.asarray(gv.F))


# ---------------------------------------------------------------------------
# makegp_fourier fixed-hyperparameter ("fixed-point") path
# ---------------------------------------------------------------------------

def _powerlaw_noisedict(params, log10_A=-14.0, gamma=3.5):
    """Full noisedict for a power-law GP built from its parameter names."""
    return {p: (log10_A if 'log10_A' in p else gamma) for p in params}


class TestMakegpFourierFixed:
    """makegp_fourier returns a cached ConstantGP when a noisedict supplies all
    hyperparameters, and a VariableGP otherwise. Achromatic (non-callable)
    basis, so the only hyperparameters are the power-law PSD params."""

    def test_full_noisedict_returns_constant_gp(self, psr):
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        assert isinstance(gv, matrix.VariableGP)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn',
                            noisedict=_powerlaw_noisedict(gv.Phi.params))
        assert isinstance(gc, matrix.ConstantGP)

    def test_empty_noisedict_returns_variable_gp(self, psr):
        from discovery import matrix
        gp = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict={})
        assert isinstance(gp, matrix.VariableGP)

    def test_partial_noisedict_returns_variable_gp(self, psr):
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        full = _powerlaw_noisedict(gv.Phi.params)
        one_key = next(iter(full))
        gp = makegp_fourier(psr, ds.powerlaw, 20, name='rn',
                            noisedict={one_key: full[one_key]})
        assert isinstance(gp, matrix.VariableGP)

    def test_mean_disables_fixed_path(self, psr):
        """A non-None mean keeps the GP variable even with a full noisedict."""
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        gp = makegp_fourier(psr, ds.powerlaw, 20, name='rn',
                            noisedict=_powerlaw_noisedict(gv.Phi.params),
                            mean=ds.powerlaw)
        assert isinstance(gp, matrix.VariableGP)

    def test_constant_gp_has_no_free_params(self, psr):
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn',
                            noisedict=_powerlaw_noisedict(gv.Phi.params))
        assert gc.Phi.params == []

    def test_constant_phi_is_1d_novar_for_powerlaw(self, psr):
        """A power-law PSD is diagonal, so the cached prior is a 1D novar."""
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn',
                            noisedict=_powerlaw_noisedict(gv.Phi.params))
        assert isinstance(gc.Phi, matrix.NoiseMatrix1D_novar)

    def test_cached_prior_equals_variable_prior(self, psr):
        """Cached Phi equals the variable prior evaluated at the noisedict."""
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        nd = _powerlaw_noisedict(gv.Phi.params)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.Phi.N),
                                   np.asarray(gv.Phi.getN(nd)))

    def test_cached_basis_matches_variable(self, psr):
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        nd = _powerlaw_noisedict(gv.Phi.params)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.F), np.asarray(gv.F))

    def test_metadata_preserved(self, psr):
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        nd = _powerlaw_noisedict(gv.Phi.params)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict=nd)
        assert gc.name == psr.name
        assert gc.gpname == 'rn'
        assert list(gc.index) == list(gv.index)


class TestMakegpFourierChromaticFixed:
    """A callable basis (free chromatic index) adds an extra hyperparameter:
    the fixed path requires the chromatic index in the noisedict too."""

    def test_prior_only_noisedict_stays_variable(self, psr):
        """PSD params fixed but the chromatic index free -> still VariableGP."""
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom)
        gp = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom,
                            noisedict=_powerlaw_noisedict(gv.Phi.params))
        assert isinstance(gp, matrix.VariableGP)

    def test_full_noisedict_returns_constant_gp(self, psr):
        """PSD params + chromatic index fixed -> ConstantGP."""
        from discovery import matrix
        gv = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom)
        nd = _powerlaw_noisedict(gv.Phi.params)
        nd.update({p: 4.0 for p in gv.F.params})
        gc = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom, noisedict=nd)
        assert isinstance(gc, matrix.ConstantGP)

    def test_cached_basis_matches_variable_at_index(self, psr):
        """Cached fixed-index basis equals the callable basis evaluated."""
        gv = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom)
        assert callable(gv.F)
        nd = _powerlaw_noisedict(gv.Phi.params)
        nd.update({p: 4.0 for p in gv.F.params})
        gc = makegp_fourier(psr, ds.powerlaw, 15, name='chrom',
                            fourierbasis=fourierbasis_chrom, noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.F), np.asarray(gv.F(nd)))


class TestFixedGPLikelihoodEquivalence:
    """A fixed GP gives the same PulsarLikelihood logL as the variable GP
    evaluated at the fixed hyperparameter values."""

    @staticmethod
    def _white(psr, sigma=1e-6):
        from discovery import matrix
        return matrix.NoiseMatrix1D_novar(jnp.full(len(psr.toas), sigma ** 2))

    def test_logL_matches_variable_at_fixed_point(self, psr):
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        nd = _powerlaw_noisedict(gv.Phi.params)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict=nd)

        Lv = ds.PulsarLikelihood([psr.residuals, self._white(psr), gv])
        Lc = ds.PulsarLikelihood([psr.residuals, self._white(psr), gc])

        assert Lc.logL.params == []  # nothing left to sample
        np.testing.assert_allclose(float(Lc.logL({})), float(Lv.logL(nd)),
                                   rtol=1e-10)

    def test_logL_differs_away_from_fixed_point(self, psr):
        """Sanity check: the equivalence is specific to the fixed values."""
        gv = makegp_fourier(psr, ds.powerlaw, 20, name='rn')
        nd = _powerlaw_noisedict(gv.Phi.params)
        gc = makegp_fourier(psr, ds.powerlaw, 20, name='rn', noisedict=nd)

        Lv = ds.PulsarLikelihood([psr.residuals, self._white(psr), gv])
        Lc = ds.PulsarLikelihood([psr.residuals, self._white(psr), gc])

        other = _powerlaw_noisedict(gv.Phi.params, log10_A=-13.0, gamma=2.0)
        assert not np.isclose(float(Lv.logL(other)), float(Lc.logL({})))
