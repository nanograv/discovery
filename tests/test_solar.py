#!/usr/bin/env python3
"""Tests for discovery.solar module"""

import pytest
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from discovery import const, matrix, solar
from discovery.solar import (
    theta_impact,
    dm_solar,
    make_solardm,
    fourierbasis_solar_dm,
    makegp_timedomain_solar_dm,
)
from discovery.signals import (
    square_exponential_kernel,
    matern_kernel,
    custom_blocked_interpolation_basis,
)

class MockPsr:
    """Mock pulsar object for testing solar wind functions."""

    def __init__(self, toas=None, freqs=None, name='J0000+0000'):
        # Default TOAs in seconds (MJD * 86400)
        self.toas = toas if toas is not None else np.array([
            55000.0 * 86400, 55001.0 * 86400, 55002.0 * 86400
        ])
        self.freqs = freqs if freqs is not None else np.array([1400.0, 1400.0, 1400.0])
        self.name = name

        # Create mock solar system ephemeris
        # planetssb shape: (n_toas, n_planets, 6) - we need Earth (index 2)
        # sunssb shape: (n_toas, 6)
        n_toas = len(self.toas)

        # Simple geometry: Earth at 1 AU in x-direction, Sun at origin
        self.planetssb = np.zeros((n_toas, 10, 6))
        # Earth position at 1 AU in x-direction (in light-seconds)
        au_light_sec = const.AU / const.c
        self.planetssb[:, 2, 0] = au_light_sec  # x-position
        self.planetssb[:, 2, 1] = 0.0  # y-position
        self.planetssb[:, 2, 2] = 0.0  # z-position

        # Sun at origin
        self.sunssb = np.zeros((n_toas, 6))

        # Pulsar position (unit vector pointing in z-direction)
        self.pos = np.array([0.0, 0.0, 1.0])
        # Replicate for each TOA
        self.pos_t = np.tile(self.pos, (n_toas, 1))
    @property
    def mintoa(self):
        return self.toas.min()

    @property
    def maxtoa(self):
        return self.toas.max()

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AU_LIGHT_SEC = const.AU / const.c   # ~499 s
AU_PC        = const.AU / const.pc  # ~4.85e-6 pc

@pytest.fixture(scope='module')
def solar_psr():
    return MockPsr()

class TestThetaImpact:
    """Tests for theta_impact function."""

    def test_theta_impact_returns_four_values(self):
        """Test that theta_impact returns four values."""
        psr = MockPsr()
        result = solar.theta_impact(psr)

        assert len(result) == 4
        theta, R_earth, b, z_earth = result
        assert theta.shape == (len(psr.toas),)
        assert R_earth.shape == (len(psr.toas),)
        assert b.shape == (len(psr.toas),)
        assert z_earth.shape == (len(psr.toas),)

    def test_theta_impact_perpendicular_geometry(self):
        """Test theta_impact with perpendicular geometry (pulsar at 90 deg from Sun)."""
        psr = MockPsr()
        theta, R_earth, b, z_earth = solar.theta_impact(psr)

        # With pulsar in z-direction and Earth in x-direction from Sun,
        # theta should be pi/2 (90 degrees)
        np.testing.assert_allclose(theta, np.pi / 2, rtol=1e-6)

        # R_earth should be approximately 1 AU in light-seconds
        au_light_sec = const.AU / const.c
        np.testing.assert_allclose(R_earth, au_light_sec, rtol=1e-6)

    def test_theta_impact_positive_values(self):
        """Test that R_earth and b are positive."""
        psr = MockPsr()
        theta, R_earth, b, z_earth = solar.theta_impact(psr)

        assert np.all(R_earth > 0)
        assert np.all(b >= 0)
        assert np.all(theta >= 0)
        assert np.all(theta <= np.pi)

class TestDmSolar:
    """Tests for dm_solar and related functions."""

    def test_dm_solar_returns_correct_shape_scalar(self):
        """Test that dm_solar returns correct shape for scalar inputs."""
        n_earth = 5.0
        theta = np.pi / 2
        r_earth = const.AU / const.c

        result = solar.dm_solar(n_earth, theta, r_earth)
        assert np.isscalar(result) or result.shape == ()

    def test_dm_solar_positive(self):
        """Test that dm_solar returns positive values for arrays."""
        n_earth = 5.0
        theta = np.linspace(0.1, np.pi - 0.1, 10)
        r_earth = const.AU / const.c

        result = solar.dm_solar(n_earth, theta, r_earth)
        assert result.shape == theta.shape
        assert np.all(result > 0)

    def test_dm_solar_scales_with_density(self):
        """Test that dm_solar scales linearly with electron density."""
        theta = np.pi / 2
        r_earth = const.AU / const.c

        dm1 = solar.dm_solar(5.0, theta, r_earth)
        dm2 = solar.dm_solar(10.0, theta, r_earth)

        np.testing.assert_allclose(dm2 / dm1, 2.0, rtol=1e-10)

    def test_dm_solar_close_approach(self):
        """Test dm_solar uses close approach approximation near pi."""
        n_earth = 5.0
        r_earth = const.AU / const.c

        # Test at threshold (pi - theta = 1e-5)
        theta_close = np.pi - 1e-6  # Should use close approximation
        theta_far = np.pi - 1e-4    # Should use regular formula

        result_close = solar.dm_solar(n_earth, theta_close, r_earth)
        result_far = solar.dm_solar(n_earth, theta_far, r_earth)

        # Both should give positive finite values
        assert np.isfinite(result_close)
        assert np.isfinite(result_far)
        assert result_close > 0
        assert result_far > 0

    def test_dm_solar_continuous_at_boundary(self):
        """Test that dm_solar is continuous at the boundary between approximations."""
        n_earth = 5.0
        r_earth = const.AU / const.c

        # Test near the boundary (pi - theta = 1e-5)
        theta_just_below = np.pi - 1e-5 - 1e-7
        theta_just_above = np.pi - 1e-5 + 1e-7

        result_below = solar.dm_solar(n_earth, theta_just_below, r_earth)
        result_above = solar.dm_solar(n_earth, theta_just_above, r_earth)

        # Results should be very close (within 1%)
        np.testing.assert_allclose(result_below, result_above, rtol=1e-2)


class TestMakeSolardm:
    """Tests for make_solardm function."""

    def test_make_solardm_returns_callable(self):
        """Test that make_solardm returns a callable function."""
        psr = MockPsr()
        solardm_func = solar.make_solardm(psr)

        assert callable(solardm_func)

    def test_make_solardm_output_shape(self):
        """Test that the returned function produces correct output shape."""
        psr = MockPsr()
        solardm_func = solar.make_solardm(psr)

        n_earth = 5.0
        result = solardm_func(n_earth)

        assert result.shape == psr.toas.shape

    def test_make_solardm_scales_linearly(self):
        """Test that output scales linearly with n_earth."""
        psr = MockPsr()
        solardm_func = solar.make_solardm(psr)

        result1 = solardm_func(5.0)
        result2 = solardm_func(10.0)

        np.testing.assert_allclose(result2 / result1, 2.0, rtol=1e-10)

    def test_make_solardm_frequency_dependence(self):
        """Test frequency-dependent scaling (proportional to 1/f^2)."""
        psr = MockPsr(
            freqs=np.array([1400.0, 2800.0, 700.0])
        )
        solardm_func = solar.make_solardm(psr)

        result = solardm_func(5.0)

        # Ratio of delays should scale as (f1/f2)^2
        # delay at 700 MHz should be 4x delay at 1400 MHz
        # This is approximate due to geometry factors
        assert result[2] > result[0]  # Lower frequency has larger delay


class TestFourierbasisSolarDm:
    """Tests for fourierbasis_solar_dm function."""

    def test_fourierbasis_solar_dm_output_shapes(self):
        """Test that fourierbasis_solar_dm returns three values with correct shapes."""
        psr = MockPsr()
        components = 10

        result = solar.fourierbasis_solar_dm(psr, components)
        assert len(result) == 3

        f, df, fmat = result

        # f should have length 2*components (repeated for sin/cos pairs)
        assert len(f) == 2 * components
        # df should be array of length 2*components
        assert len(df) == 2 * components
        # fmat should have shape (n_toas, 2*components)
        assert fmat.shape == (len(psr.toas), 2 * components)


class TestMakegpTimeDomainSolarDm:

    # ---- fixtures -----------------------------------------------------------

    @pytest.fixture
    def sq_exp_cov(self):
        return square_exponential_kernel(log10_sigma_sq_exp=-7., log10_ell=8.)

    @pytest.fixture
    def matern_cov(self):
        return matern_kernel(log10_sigma_matern=-7., log10_ell=8., nu=1.5)

    def _make_custom_basis(self, solar_psr, n_nodes=12):
        """Build a custom node-based Umat and node array for solar_psr."""
        # `solar_psr.toas` are stored in seconds since MJD 0, while
        # `custom_blocked_interpolation_basis` expects node positions in MJD.
        start_mjd = solar_psr.mintoa / 86400.0
        end_mjd = solar_psr.maxtoa / 86400.0
        nodes_mjd = np.linspace(start_mjd, end_mjd, n_nodes)
        Umat_raw, nodes_s = custom_blocked_interpolation_basis(
            solar_psr.toas, nodes_mjd, kind='linear'
        )
        return Umat_raw, nodes_s

    # ---- auto-quantized basis -----------------------------------------------

    def test_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        assert isinstance(gp, matrix.VariableGP)

    def test_basis_shape_auto(self, solar_psr, sq_exp_cov):
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        assert gp.F.shape[0] == len(solar_psr.toas)
        assert gp.F.shape[1] >= 1

    def test_param_names_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30, name='sw_gp'
        )
        for p in gp.Phi.params:
            assert p.startswith(solar_psr.name), f"Param {p!r} missing pulsar prefix"
            assert 'sw_gp' in p

    def test_phi_square_matrix_auto(self, solar_psr, sq_exp_cov):
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        n_bins = gp.F.shape[1]
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        assert phi.shape == (n_bins, n_bins)

    def test_phi_symmetric_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi).T, atol=1e-12)

    def test_phi_positive_definite_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        eigvals = np.linalg.eigvalsh(np.asarray(phi))
        assert np.all(eigvals > 0)

    def test_index_attribute_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        assert len(gp.index) == 1
        key = next(iter(gp.index))
        assert key.startswith(solar_psr.name)

    def test_solar_dm_scaling_in_basis_auto(self, solar_psr, sq_exp_cov):
        """Columns of F reflect (freq-dependent) solar wind DM scaling."""
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        theta, r_earth, _, _  = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        expected_scale = dm_sol * 4.148808e3 / solar_psr.freqs**2

        F = np.asarray(gp.F)
        # For each active bin find two TOAs and check the weight ratio matches
        # the ratio of their DM scalings.
        for col in range(F.shape[1]):
            rows = np.where(np.abs(F[:, col]) > 0)[0]
            if len(rows) < 2:
                continue
            i, j = rows[0], rows[1]
            if expected_scale[j] == 0:
                continue
            ratio_F  = F[i, col] / F[j, col]
            ratio_dm = expected_scale[i] / expected_scale[j]
            np.testing.assert_allclose(ratio_F, ratio_dm, rtol=1e-10)
            break

    def test_different_kernels_different_phi_auto(self, sq_exp_cov, matern_cov):
        dt = 86400 * 30
        psr = MockPsr(
            toas=np.linspace(55000.0 * 86400, 55150.0 * 86400, 60),
            freqs=np.full(60, 1400.0)
        )
        gp_sq = makegp_timedomain_solar_dm(psr, sq_exp_cov, dt=dt)
        gp_mt = makegp_timedomain_solar_dm(psr, matern_cov, dt=dt)
        assert gp_sq.F.shape == gp_mt.F.shape

        def _eval(gp):
            params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(3.0)
                      for p in gp.Phi.params}
            return gp.Phi.getN(params)

        phi_sq = _eval(gp_sq)
        phi_mt = _eval(gp_mt)
        mask = ~np.eye(phi_sq.shape[0], dtype=bool)
        diff = np.abs(np.asarray(phi_sq)[mask] - np.asarray(phi_mt)[mask])
        scale = np.abs(np.asarray(phi_sq)[mask]) + 1e-300
        assert np.max(diff / scale) > 0.01, \
            "sq-exp and Matérn kernels produced nearly identical covariance matrices"

    # ---- custom basis (Umat + nodes) ----------------------------------------

    def test_returns_variable_gp_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        assert isinstance(gp, matrix.VariableGP)

    def test_basis_shape_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=10)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        assert gp.F.shape[0] == len(solar_psr.toas)
        assert gp.F.shape[1] == Umat.shape[1]

    def test_phi_square_matrix_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=8)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        n_nodes = gp.F.shape[1]
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        assert phi.shape == (n_nodes, n_nodes)

    def test_phi_symmetric_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi).T, atol=1e-12)

    def test_phi_positive_definite_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        eigvals = np.linalg.eigvalsh(np.asarray(phi))
        assert np.all(eigvals > 0)

    def test_custom_umat_has_solar_dm_columns(self, solar_psr, sq_exp_cov):
        """Each column of gp.F should be scaled by the solar DM weight."""
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        theta, r_earth, _, _  = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        dt_DM  = dm_sol * 4.148808e3 / solar_psr.freqs**2

        F = np.asarray(gp.F)
        # The stored F equals Umat * dt_DM[:, None]; verify this ratio per col
        Umat_expected = Umat * dt_DM[:, None]
        np.testing.assert_allclose(F, Umat_expected, rtol=1e-10)

    def test_nodes_required_with_custom_umat(self, solar_psr, sq_exp_cov):
        """Providing Umat without nodes should raise AssertionError."""
        Umat, _ = self._make_custom_basis(solar_psr)
        with pytest.raises(AssertionError, match="nodes"):
            makegp_timedomain_solar_dm(
                solar_psr, sq_exp_cov, Umat=Umat, nodes=None
            )

    def test_common_param_custom(self, solar_psr, sq_exp_cov):
        """Parameter in common list should not carry the pulsar-name prefix."""
        import inspect as _inspect
        args = _inspect.getfullargspec(sq_exp_cov).args
        sigma_arg = next(a for a in args if 'sigma' in a)
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes,
            name='sw_gp', common=[sigma_arg]
        )
        assert sigma_arg in gp.Phi.params, \
            f"Expected bare common param {sigma_arg!r} in {gp.Phi.params}"

    def test_auto_and_custom_basis_same_node_count(self, solar_psr, sq_exp_cov):
        """Custom basis with n_nodes nodes gives gp.F with n_nodes columns."""
        for n_nodes in [6, 10, 15]:
            Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=n_nodes)
            gp = makegp_timedomain_solar_dm(
                solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
            )
            assert gp.F.shape[1] == Umat.shape[1]

    # ---- fixed-hyperparameter ("fixed-point") path --------------------------

    @staticmethod
    def _full_noisedict(params):
        """Full noisedict for the square-exponential covariance params."""
        return {p: (-7.0 if 'sigma' in p else 8.0) for p in params}

    def test_empty_noisedict_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict={})
        assert isinstance(gp, matrix.VariableGP)

    def test_full_noisedict_returns_constant_gp(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        gc = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30,
            noisedict=self._full_noisedict(gv.Phi.params)
        )
        assert isinstance(gc, matrix.ConstantGP)
        assert isinstance(gc.Phi, matrix.NoiseMatrix2D_novar)

    def test_partial_noisedict_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        full = self._full_noisedict(gv.Phi.params)
        one_key = next(iter(full))
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30,
            noisedict={one_key: full[one_key]}
        )
        assert isinstance(gp, matrix.VariableGP)

    def test_cached_covariance_matches_variable(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.Phi.N),
                                   np.asarray(gv.Phi.getN(nd)))

    def test_cached_basis_matches_variable(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.F), np.asarray(gv.F))

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_solar_wind_pipeline(self):
        """Test the complete solar wind modeling pipeline."""
        # Create a mock pulsar with multiple TOAs
        psr = MockPsr(
            toas=np.linspace(55000.0 * 86400, 55100.0 * 86400, 50),
            freqs=np.full(50, 1400.0)
        )

        # Calculate solar geometry
        theta, R_earth, b, z_earth = solar.theta_impact(psr)
        assert theta.shape == (50,)

        # Calculate DM contribution
        dm = solar.dm_solar(5.0, theta, R_earth)
        assert dm.shape == (50,)
        assert np.all(dm > 0)

        # Create solar DM function
        solardm_func = solar.make_solardm(psr)
        dm_delays = solardm_func(5.0)
        assert dm_delays.shape == (50,)

    def test_gp_construction_pipeline(self):
        """Test GP construction with solar wind geometry."""
        psr = MockPsr(
            toas=np.linspace(55000.0 * 86400, 55010.0 * 86400, 20),
            freqs=np.full(20, 1400.0)
        )

        # Create time-domain GP
        def exponential_cov(tau, log10_sigma, log10_ell):
            return 10**(2 * log10_sigma) * jnp.exp(-tau / 10**log10_ell)

        gp = solar.makegp_timedomain_solar_dm(psr, exponential_cov, dt=86400.0)

        # Check GP structure
        assert isinstance(gp, matrix.VariableGP)
        assert hasattr(gp, 'Phi')  # Covariance matrix
        assert hasattr(gp, 'F')    # Basis matrix

        # Basis should have correct shape
        # (n_toas, n_bins) where n_bins depends on quantization
        assert gp.F.shape[0] == len(psr.toas)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
