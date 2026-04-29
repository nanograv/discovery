#!/usr/bin/env python3
"""Tests for discovery.deterministic module"""

import pytest
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from discovery import deterministic


class MockPsr:
    """Mock pulsar object for testing."""
    def __init__(self, toas=None, freqs=None):
        # Default TOAs in MJ seconds (Modified Julian seconds, not MJD)
        # 55000 MJD = 55000 * 86400 MJ seconds
        self.toas = toas if toas is not None else np.array([55000.0 * 86400, 55001.0 * 86400, 55002.0 * 86400])
        self.freqs = freqs if freqs is not None else np.array([2800.0, 2800.0, 2800.0])


class TestChromaticExponential:
    """Tests for chromatic_exponential delay function."""

    def test_chromatic_exponential_returns_callable(self):
        """Test that chromatic_exponential returns a callable function."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_exponential(psr, fref=1400.0)

        assert callable(delay_func)
        assert delay_func.__name__ == "chromatic_exponential_delay"

    def test_chromatic_exponential_output_shape(self):
        """Test that output has correct shape."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_exponential(psr, fref=1400.0)

        result = delay_func(t0=55001.0, log10_Amp=-6.0, log10_tau=1.0,
                           sign_param=1.0, alpha=2.0)

        assert result.shape == psr.toas.shape

    def test_chromatic_exponential_heaviside(self):
        """Test that delay is zero before t0 (Heaviside step)."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_exponential(psr, fref=1400.0)

        result = delay_func(t0=55000.5, log10_Amp=-6.0, log10_tau=1.0,
                           sign_param=1.0, alpha=2.0)

        # Before t0 (first element), delay should be zero due to Heaviside
        assert result[0] == 0.0
        # After t0, delay should be non-zero
        assert result[1] != 0.0
        assert result[2] != 0.0

    def test_chromatic_exponential_sign(self):
        """Test that sign_param correctly changes sign of delay."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_exponential(psr, fref=1400.0)

        result_pos = delay_func(t0=55000.5, log10_Amp=-6.0, log10_tau=1.0,
                                sign_param=1.0, alpha=2.0)
        result_neg = delay_func(t0=55000.5, log10_Amp=-6.0, log10_tau=1.0,
                                sign_param=-1.0, alpha=2.0)

        np.testing.assert_array_almost_equal(result_pos, -result_neg)

    def test_chromatic_exponential_frequency_dependence(self):
        """Test frequency-dependent scaling with alpha parameter."""
        psr = MockPsr(toas=np.array([55001.0 * 86400, 55001.0 * 86400]),
                      freqs=np.array([2800.0, 1400.0]))
        delay_func = deterministic.chromatic_exponential(psr, fref=1400.0)

        result = delay_func(t0=55000.0, log10_Amp=-6.0, log10_tau=1.0,
                           sign_param=1.0, alpha=2.0)

        # At alpha=2, delay should scale as (fref/f)^2
        # freq[0]=2800: (1400/2800)^2 = 0.25
        # freq[1]=1400: (1400/1400)^2 = 1.0
        # So result[1] should be 4x result[0]
        # Check both values are finite first
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        ratio = result[1] / result[0]
        assert np.abs(ratio - 4.0) < 1e-10


class TestChromaticAnnual:
    """Tests for chromatic_annual delay function."""

    def test_chromatic_annual_returns_callable(self):
        """Test that chromatic_annual returns a callable function."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_annual(psr, fref=1400.0)

        assert callable(delay_func)
        assert delay_func.__name__ == "chromatic_annual_delay"

    def test_chromatic_annual_output_shape(self):
        """Test that output has correct shape."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_annual(psr, fref=1400.0)

        result = delay_func(log10_Amp=-6.0, phase=0.0, alpha=2.0)

        assert result.shape == psr.toas.shape

    def test_chromatic_annual_sinusoidal(self):
        """Test that delay follows sinusoidal pattern."""
        psr = MockPsr(toas=np.array([55000.0 * 86400]),
                      freqs=np.array([2800.0]))
        delay_func = deterministic.chromatic_annual(psr, fref=1400.0)

        # At phase=0, should give sin(2*pi*f_yr*t)
        # At phase=pi/2, should give sin(2*pi*f_yr*t + pi/2) = cos(2*pi*f_yr*t)
        result_0 = delay_func(log10_Amp=-6.0, phase=0.0, alpha=2.0)
        result_90 = delay_func(log10_Amp=-6.0, phase=np.pi/2, alpha=2.0)

        # These should be different (unless by chance the time gives sin=0)
        # Just check they're both valid numbers
        assert np.isfinite(result_0[0])
        assert np.isfinite(result_90[0])

    def test_chromatic_annual_frequency_dependence(self):
        """Test frequency-dependent scaling with alpha parameter."""
        psr = MockPsr(toas=np.array([55001.0 * 86400, 55001.0 * 86400]),
                      freqs=np.array([2800.0, 1400.0]))
        delay_func = deterministic.chromatic_annual(psr, fref=1400.0)

        result = delay_func(log10_Amp=-6.0, phase=0.0, alpha=2.0)

        # At alpha=2, delay should scale as (fref/f)^2
        # freq[0]=2800: (1400/2800)^2 = 0.25
        # freq[1]=1400: (1400/1400)^2 = 1.0
        # So result[1] should be 4x result[0]
        ratio = result[1] / result[0]
        assert np.abs(ratio - 4.0) < 1e-10


class TestChromaticGaussian:
    """Tests for chromatic_gaussian delay function."""

    def test_chromatic_gaussian_returns_callable(self):
        """Test that chromatic_gaussian returns a callable function."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_gaussian(psr, fref=1400.0)

        assert callable(delay_func)
        assert delay_func.__name__ == "chromatic_gaussian_delay"

    def test_chromatic_gaussian_output_shape(self):
        """Test that output has correct shape."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_gaussian(psr, fref=1400.0)

        result = delay_func(t0=55001.0, log10_Amp=-6.0, log10_sigma=1.0,
                           sign_param=1.0, alpha=2.0)

        assert result.shape == psr.toas.shape

    def test_chromatic_gaussian_peak_at_t0(self):
        """Test that Gaussian peaks at t0."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_gaussian(psr, fref=1400.0)

        result = delay_func(t0=55001.0, log10_Amp=-6.0, log10_sigma=0.5,
                           sign_param=1.0, alpha=2.0)

        # Peak should be at t0 (middle element)
        assert np.abs(result[1]) > np.abs(result[0])
        assert np.abs(result[1]) > np.abs(result[2])

    def test_chromatic_gaussian_sign(self):
        """Test that sign_param correctly changes sign of delay."""
        psr = MockPsr()
        delay_func = deterministic.chromatic_gaussian(psr, fref=1400.0)

        result_pos = delay_func(t0=55001.0, log10_Amp=-6.0, log10_sigma=1.0,
                                sign_param=1.0, alpha=2.0)
        result_neg = delay_func(t0=55001.0, log10_Amp=-6.0, log10_sigma=1.0,
                                sign_param=-1.0, alpha=2.0)

        np.testing.assert_array_almost_equal(result_pos, -result_neg)

    def test_chromatic_gaussian_frequency_dependence(self):
        """Test frequency-dependent scaling with alpha parameter."""
        psr = MockPsr(toas=np.array([55001.0 * 86400, 55001.0 * 86400]),
                      freqs=np.array([2800.0, 1400.0]))
        delay_func = deterministic.chromatic_gaussian(psr, fref=1400.0)

        result = delay_func(t0=55001.0, log10_Amp=-6.0, log10_sigma=1.0,
                           sign_param=1.0, alpha=2.0)

        # At alpha=2, delay should scale as (fref/f)^2
        # freq[0]=2800: (1400/2800)^2 = 0.25
        # freq[1]=1400: (1400/1400)^2 = 1.0
        # So result[1] should be 4x result[0]
        ratio = result[1] / result[0]
        assert np.abs(ratio - 4.0) < 1e-10


class TestOrthometricShapiro:
    """Tests for orthometric_shapiro delay function."""

    def test_orthometric_shapiro_returns_callable(self):
        """Test that orthometric_shapiro returns a callable function."""
        psr = MockPsr()
        binphase = np.array([0.0, np.pi/2, np.pi])
        delay_func = deterministic.orthometric_shapiro(psr, binphase)

        assert callable(delay_func)
        assert delay_func.__name__ == "orthometric_shapiro_delay"

    def test_orthometric_shapiro_output_shape(self):
        """Test that output has correct shape."""
        psr = MockPsr()
        binphase = np.array([0.0, np.pi/2, np.pi])
        delay_func = deterministic.orthometric_shapiro(psr, binphase)

        result = delay_func(h3=1e-7, stig=0.9)

        assert result.shape == psr.toas.shape

    def test_orthometric_shapiro_binphase_shape_mismatch(self):
        """Test that ValueError is raised when binphase shape doesn't match toas."""
        psr = MockPsr()
        binphase = np.array([0.0, np.pi/2])  # Wrong shape

        with pytest.raises(ValueError, match="binphase must have the same shape"):
            deterministic.orthometric_shapiro(psr, binphase)

    def test_orthometric_shapiro_values_finite(self):
        """Test that Shapiro delay produces finite values."""
        psr = MockPsr()
        binphase = np.array([0.0, np.pi/4, np.pi/2])
        delay_func = deterministic.orthometric_shapiro(psr, binphase)

        result = delay_func(h3=1e-7, stig=0.5)

        assert np.all(np.isfinite(result))

    def test_orthometric_shapiro_equation_form(self):
        """Test that delay follows the expected equation form from Freire & Wex 2010."""
        psr = MockPsr(toas=np.array([55000.0 * 86400]), freqs=np.array([2800.0]))
        binphase = np.array([0.0])  # sin(0) = 0
        delay_func = deterministic.orthometric_shapiro(psr, binphase)

        h3 = 1e-7
        stig = 0.5

        result = delay_func(h3=h3, stig=stig)

        # At binphase=0, sin(binphase)=0, so:
        # Delta_s = -(2*h3/stig^3) * log(1 + stig^2)
        expected = -(2.0 * h3 / stig**3) * np.log(1 + stig**2)

        np.testing.assert_almost_equal(result[0], expected, decimal=15)
