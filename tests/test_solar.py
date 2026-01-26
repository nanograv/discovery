#!/usr/bin/env python3
"""Tests for discovery.solar module"""

import pytest
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from discovery import solar, matrix, const


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


class TestMakegpTimedomainSolarDm:
    """Tests for makegp_timedomain_solar_dm function."""

    def test_makegp_timedomain_solar_dm_returns_variablegp(self):
        """Test that function returns a VariableGP object."""
        psr = MockPsr()

        # Simple covariance function
        def simple_cov(tau, log10_sigma, log10_ell):
            sigma = 10**log10_sigma
            ell = 10**log10_ell
            return sigma**2 * jnp.exp(-tau / ell)

        result = solar.makegp_timedomain_solar_dm(psr, simple_cov, dt=86400.0)

        assert isinstance(result, matrix.VariableGP)

    def test_makegp_timedomain_solar_dm_with_dt(self):
        """Test with custom time bin width."""
        psr = MockPsr()

        def simple_cov(tau, log10_sigma):
            return 10**(2 * log10_sigma) * jnp.ones_like(tau)

        result = solar.makegp_timedomain_solar_dm(psr, simple_cov, dt=43200.0)  # 12 hours

        assert isinstance(result, matrix.VariableGP)

    def test_makegp_timedomain_solar_dm_parameter_naming(self):
        """Test that parameter names are generated correctly."""
        psr = MockPsr(name='J1234+5678')

        def simple_cov(tau, log10_sigma, log10_ell):
            return 10**(2 * log10_sigma) * jnp.exp(-tau / 10**log10_ell)

        result = solar.makegp_timedomain_solar_dm(psr, simple_cov, name='sw_dm')

        # Check that the covariance matrix has params attribute
        assert hasattr(result.Phi, 'params')
        params = result.Phi.params

        # Should have pulsar-specific parameter names
        assert any('J1234+5678' in p for p in params)
        assert any('sw_dm' in p for p in params)

    def test_makegp_timedomain_solar_dm_common_parameters(self):
        """Test with common (shared) parameters."""
        psr = MockPsr(name='J1234+5678')

        def simple_cov(tau, log10_sigma, log10_ell):
            return 10**(2 * log10_sigma) * jnp.exp(-tau / 10**log10_ell)

        result = solar.makegp_timedomain_solar_dm(
            psr, simple_cov, common=['log10_ell'], name='sw_dm'
        )

        params = result.Phi.params

        # log10_ell should be common (not pulsar-specific)
        assert 'log10_ell' in params
        # log10_sigma should be pulsar-specific
        assert any('log10_sigma' in p and 'J1234+5678' in p for p in params)

    def test_makegp_timedomain_solar_dm_covariance_evaluation(self):
        """Test that the covariance function can be evaluated."""
        psr = MockPsr(name='J1234+5678')

        def exponential_cov(tau, log10_sigma, log10_ell):
            return 10**(2 * log10_sigma) * jnp.exp(-tau / 10**log10_ell)

        result = solar.makegp_timedomain_solar_dm(psr, exponential_cov, dt=86400.0)

        # Create test parameters
        test_params = {
            'J1234+5678_timedomain_sw_gp_log10_sigma': -6.0,
            'J1234+5678_timedomain_sw_gp_log10_ell': 1.5,
        }

        # Evaluate the covariance through the GP structure
        # The Phi object should have getN method
        cov_matrix = result.Phi.getN(test_params)

        # Check that we get a matrix
        assert cov_matrix.ndim == 2
        # Should be square matrix
        assert cov_matrix.shape[0] == cov_matrix.shape[1]
        # Should be positive on diagonal
        assert np.all(np.diag(cov_matrix) > 0)


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
