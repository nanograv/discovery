#!/usr/bin/env python3
"""Tests for discovery matrix operations"""

import pytest
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import discovery as ds
from discovery import matrix


class TestWoodburyKernel:
    def test_WoodburyKernel_varNP_vs_varP(self):
        """
        Test that WoodburyKernel_varNP with fixed white noise parameters
        produces the same results as WoodburyKernel_varP.

        This verifies that the varNP implementation correctly reduces to varP
        when the white noise parameters are held constant.
        """
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create test data
        n_data = 100
        n_basis = 10

        y = np.random.randn(n_data)
        F = np.random.randn(n_data, n_basis)

        # Create base white noise diagonal matrix
        N_base = np.random.uniform(0.1, 2.0, n_data)

        # Fixed efac value for the test (non-trivial to avoid masking issues)
        fixed_efac = 1.3

        # Create a function that returns N (for varNP)
        # This simulates a white noise parameter that could vary
        wn_param_name = 'test_efac'
        def getN(params):
            # When called with the fixed value, should return N_base * efac**2
            # For this test, we'll use a simple scaling factor
            efac = params.get(wn_param_name, fixed_efac)
            return N_base * efac**2
        getN.params = [wn_param_name]

        # Create P_var for varying red noise parameters
        # Using a simple diagonal covariance for testing
        # P is typically the prior covariance matrix for GP coefficients
        rn_param_name = 'test_log10_A'

        # Create a simple variable diagonal P matrix
        def getP_diag(params):
            log10_A = params[rn_param_name]
            A = 10**log10_A
            return jnp.ones(n_basis) * A
        getP_diag.params = [rn_param_name]

        P_var = matrix.NoiseMatrix1D_var(getP_diag)

        # Create N_var object for varNP
        N_var = matrix.NoiseMatrix1D_var(getN)

        # Create N object for varP (fixed white noise at the scaled value)
        N_fixed = matrix.NoiseMatrix1D_novar(N_base * fixed_efac**2)

        # Construct both kernel models
        kernel_varNP = matrix.WoodburyKernel_varNP(N_var, F, P_var)
        kernel_varP = matrix.WoodburyKernel_varP(N_fixed, F, P_var)

        # Make kernelproduct functions
        kp_varNP = kernel_varNP.make_kernelproduct(y)
        kp_varP = kernel_varP.make_kernelproduct(y)

        # Test with a fixed white noise value (efac = 1.3, a non-trivial value)
        # This should make varNP behave identically to varP
        # Use red noise amplitude comparable to white noise for meaningful signal
        # Use non-trivial values to avoid masking issues (no 0.0 or 1.0)
        test_params_varNP = {
            wn_param_name: fixed_efac,  # Fixed white noise (1.3)
            rn_param_name: 0.3  # log10(A) = 0.3 => A ~ 2.0, comparable to white noise
        }

        # For varP, we only need the red noise parameter
        test_params_varP = {
            rn_param_name: 0.3
        }

        # Compute log-likelihood from both models
        logL_varNP = kp_varNP(test_params_varNP)
        logL_varP = kp_varP(test_params_varP)

        # They should be identical (within numerical precision)
        assert np.isclose(logL_varNP, logL_varP, rtol=1e-10, atol=1e-10), \
            f"varNP and varP should produce identical results when white noise is fixed. " \
            f"Got varNP={logL_varNP}, varP={logL_varP}, diff={abs(logL_varNP - logL_varP)}"

        # Test with a different red noise parameter value
        test_params_varNP_2 = {
            wn_param_name: fixed_efac,  # Keep white noise fixed at 1.3
            rn_param_name: 0.8  # log10(A) = 0.8 => A ~ 6.3
        }
        test_params_varP_2 = {
            rn_param_name: 0.8
        }

        logL_varNP_2 = kp_varNP(test_params_varNP_2)
        logL_varP_2 = kp_varP(test_params_varP_2)

        assert np.isclose(logL_varNP_2, logL_varP_2, rtol=1e-10, atol=1e-10), \
            f"varNP and varP should produce identical results for different parameters. " \
            f"Got varNP={logL_varNP_2}, varP={logL_varP_2}, diff={abs(logL_varNP_2 - logL_varP_2)}"

        # Verify that the two likelihood values are different for different parameters
        assert not np.isclose(logL_varNP, logL_varNP_2, rtol=1e-3, atol=0.01), \
            f"Likelihood should change with different red noise parameters. " \
            f"Got logL_1={logL_varNP}, logL_2={logL_varNP_2}, diff={abs(logL_varNP - logL_varNP_2)}"

        # Test with JIT compilation
        jit_kp_varNP = jax.jit(kp_varNP)
        jit_kp_varP = jax.jit(kp_varP)

        logL_varNP_jit = jit_kp_varNP(test_params_varNP)
        logL_varP_jit = jit_kp_varP(test_params_varP)

        assert np.isclose(logL_varNP_jit, logL_varP_jit, rtol=1e-10, atol=1e-10), \
            f"JIT-compiled varNP and varP should produce identical results. " \
            f"Got varNP={logL_varNP_jit}, varP={logL_varP_jit}, diff={abs(logL_varNP_jit - logL_varP_jit)}"

        # Verify JIT gives same results as non-JIT
        assert np.isclose(logL_varNP, logL_varNP_jit, rtol=1e-10, atol=1e-10), \
            "JIT and non-JIT varNP should give identical results"
        assert np.isclose(logL_varP, logL_varP_jit, rtol=1e-10, atol=1e-10), \
            "JIT and non-JIT varP should give identical results"
