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


class TestSignals:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    psr_files = [
        data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
        data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
    ]
    psrs = [ds.Pulsar.read_feather(psr) for psr in psr_files]

    psr_rn_params = [f"{psrs[0].name}_red_noise_log10_A", f"{psrs[0].name}_red_noise_gamma", f"{psrs[1].name}_red_noise_log10_A", f"{psrs[1].name}_red_noise_gamma"]

    # create fake parameter dict for testing
    fake_params = {key: np.random.rand()*3 - 13 for key in psr_rn_params if 'log10_A' in key}
    fake_params = {**fake_params, **{key: np.random.rand()*3 for key in psr_rn_params if 'gamma' in key}}

    @pytest.mark.integration
    def test_makecommongp_fourier_basis_construction(self):
        tspans = [ds.getspan([psr]) for psr in self.psrs]
        # make GP for both pulsars at once

        gp = ds.makecommongp_fourier(self.psrs, ds.powerlaw, 14, T=tspans, name="red_noise")

        # make two separate GPs
        gp_psr1 = ds.makegp_fourier(self.psrs[0], ds.powerlaw, 14, T=tspans[0], name="red_noise")
        gp_psr2 = ds.makegp_fourier(self.psrs[1], ds.powerlaw, 14, T=tspans[1], name="red_noise")

        # check that bases are correct.
        npt.assert_allclose(gp.F[0], gp_psr1.F)
        npt.assert_allclose(gp.F[1], gp_psr2.F)

        # check that noise matrices are correct
        npt.assert_allclose(gp.Phi.getN(self.fake_params), np.vstack([gp_psr1.Phi.getN(self.fake_params), gp_psr2.Phi.getN(self.fake_params)]))

        # now make the bases the same, giving a single tspan
        # make GP for both pulsars at once
        tspan_total = ds.getspan(self.psrs)
        gp = ds.makecommongp_fourier(self.psrs, ds.powerlaw, 14, T=tspan_total, name="gw")

        # make two separate GPs
        gp_psr1 = ds.makegp_fourier(self.psrs[0], ds.powerlaw, 14, T=tspan_total, name="gw")
        gp_psr2 = ds.makegp_fourier(self.psrs[1], ds.powerlaw, 14, T=tspan_total, name="gw")
        # check that bases are correct.
        npt.assert_allclose(gp.F[0], gp_psr1.F)
        npt.assert_allclose(gp.F[1], gp_psr2.F)

        # check that parameters are what they should be
        gp = ds.makecommongp_fourier(self.psrs, ds.powerlaw, 14, T=tspans, name="red_noise")
        expected_pars = set(self.psr_rn_params)
        assert set(gp.Phi.params) == expected_pars

        # check that common parameters are included
        common = ["crn_log10_A", "crn_gamma"]
        powerlaw = ds.makepowerlaw_crn(14)
        gp = ds.makecommongp_fourier(self.psrs, powerlaw, 30, T=tspans, common=common, name="red_noise")
        expected_pars = set(self.psr_rn_params + common)
        assert set(gp.Phi.params) == expected_pars
