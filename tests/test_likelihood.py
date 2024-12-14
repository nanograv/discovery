#!/usr/bin/env python3
"""Tests for discovery likelihood"""

import operator
from functools import reduce
from pathlib import Path

import discovery as ds
import jax
import pytest


class TestLikelihood:
    @pytest.mark.integration
    def test_compare_enterprise(self):
        # The directory containing the pulsar feather files should be parallel to the tests directory
        data_dir = Path(__file__).resolve().parent.parent / "data"

        # Choose two pulsars for reproducibility
        psr_files = [
            data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
            data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
        ]

        # Construct a list of Pulsar objects
        psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

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
                        ds.makegp_fourier(psrs[ii], ds.powerlaw, 30, T=tspan, name="red_noise"),
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
