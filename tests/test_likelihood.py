#!/usr/bin/env python3
"""Tests for discovery likelihood"""

import operator
from functools import reduce
from pathlib import Path

import discovery as ds
import jax
import pytest
import numpy as np
import numpy.testing as npt


class TestLikelihood:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    psr_files = [
        data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
        data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
    ]

    # Construct a list of Pulsar objects
    psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

    # a simple test pulsar for testing the likelihood
    Ntoas = 10
    test_psr = psrs[0]
    test_psr.toas = np.arange(0, Ntoas)
    toaerr = np.random.rand()
    test_psr.toaerrs = np.ones_like(test_psr.toas) * toaerr
    test_psr.residuals = np.random.randn(len(test_psr.toas))
    test_psr.name = 'TEST'
    test_psr.backend_flags = np.array(['test']*len(test_psr.toas))
    test_psr.freqs = np.ones_like(test_psr.toas) * 1400
    fake_noisedict = {'TEST_test_efac': 1, 'TEST_test_log10_t2equad': -30}

    # let's reset this just to be safe.
    psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

    @pytest.mark.integration
    def test_single_psr_likelihood_varP_rn(self):
        test_psr = TestLikelihood.test_psr
        fake_model = ds.PulsarLikelihood([test_psr.residuals,
                                          ds.makenoise_measurement(
                                              test_psr,
                                              TestLikelihood.fake_noisedict),
                                          ds.makegp_fourier(test_psr,
                                                            k

        rho = np.random.rand()

        Nmat = TestLikelihood.toaerr**2 * np.eye(TestLikelihood.Ntoas)
        FPhiF = fake_model.N.F @ np.eye(4)*rho**2 @ fake_model.N.F.T
        Cmat = Nmat + FPhiF

        ll = -0.5 * test_psr.residuals @ np.linalg.solve(
            Cmat, test_psr.residuals) - 0.5 * np.linalg.slogdet(Cmat)[1]

        # test marginalized likelihood
        npt.assert_allclose(fake_model.logL(
            {'TEST_test_fourier_log10_rho(2)': np.array([np.log10(rho),
                                                         np.log10(rho)])}), ll)

        # test clogL
        c_vec = np.random.randn(4) * rho
        Phi = np.eye(4) * rho**2

        pardict = {'TEST_test_fourier_log10_rho(2)': np.array([np.log10(rho),
                                                               np.log10(rho)]),
                   'TEST_test_fourier_coefficients(4)': c_vec}
        cloglval = fake_model.clogL(pardict)

        # subtract model, r = deltaT - Fc
        rvals = (test_psr.residuals - fake_model.N.F @ c_vec)

        # -1/2 * r^T N^-1 r - 1/2 logdet(N) - 1/2 c^T Phi^-1 c - 1/2 logdet(Phi)
        clogl_direct = -0.5 * rvals.T @ np.linalg.solve(Nmat,  rvals) - 0.5 * np.linalg.slogdet(Nmat)[1] + \
                       -0.5 * \
            c_vec.T @ np.linalg.solve(Phi, c_vec) - \
            0.5 * np.linalg.slogdet(Phi)[1]
        npt.assert_allclose(cloglval, clogl_direct)

    @pytest.mark.integration
    def test_compare_enterprise(self):
        # The directory containing the pulsar feather files should be parallel to the tests directory

        # Choose two pulsars for reproducibility
        psrs = TestLikelihood.psrs

        # Get the timespan
        tspan = ds.getspan(psrs)

        # Construct the discovery global likelihood for CURN
        gl = ds.GlobalLikelihood(
            (
                ds.PulsarLikelihood(
                    [
                        psrs[ii].residuals,
                        ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                        ds.makegp_timing(psrs[ii]),
                        ds.makegp_fourier(
                            psrs[ii], ds.powerlaw, 30, T=tspan, name="red_noise"),
                        ds.makegp_fourier(
                            psrs[ii], ds.powerlaw, 14, T=tspan, common=["gw_log10_A", "gw_gamma"], name="gw"
                        ),
                    ]
                )
                for ii in range(len(psrs))
            )
        )

        # Get the jitted discovery log-likelihood
        jlogl = jax.jit(gl.logL)

        # Set parameters to feed likelihood
        initial_position = {
            "B1855+09_red_noise_gamma": 6.041543719234379,
            "B1855+09_red_noise_log10_A": -14.311870465932676,
            "B1953+29_red_noise_gamma": 2.037363188329115,
            "B1953+29_red_noise_log10_A": -16.748409409147907,
            "gw_gamma": 1.6470255693110927,
            "gw_log10_A": -14.236953140132435,
        }

        # Enterprise log-likelihood for this choice of parameters
        enterprise_ll = 145392.54369264

        # Find the difference between enterprise and discovery likelihoods
        ll_difference = enterprise_ll - jlogl(initial_position)

        # There is a constant offset of ~ -52.4
        offset = -52.4 - 5866.5585968

        # Choose the absolute tolerance
        atol = 0.1

        assert jax.numpy.abs(ll_difference - offset) <= atol
