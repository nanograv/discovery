#!/usr/bin/env python3
"""Tests for discovery likelihood"""

from pathlib import Path
import pytest

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import discovery as ds


class TestLikelihood:
    def _singlepsr_likelihoods(self, psr):
        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement_simple(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr),
                                   ds.makegp_ecorr(psr)])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict)])

        mdl = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True)])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict, ecorr=True),
                                   ds.makegp_timing(psr, svd=True)]), p0

        mdl = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr),
                                   ds.makegp_ecorr(psr),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, ecorr=True),
                                   ds.makegp_timing(psr, svd=True, variable=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')]), p0

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])

        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.partial(ds.powerlaw, gamma=4.33), components=30, name='rednoise')])

        p0 = {f'{psr.name}_rednoise_log10_rho(30)': 1e-6 * np.random.randn(30)}
        yield ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   ds.makegp_fourier(psr, ds.freespectrum, components=30, name='rednoise')]), p0

    def _multipsr_likelihoods(self, psrs):
        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])
                                  for psr in psrs])

        T = ds.getspan(psrs)
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name='rednoise'),
                                                       ds.makegp_fourier(psr, ds.powerlaw, components=14, T=T, name='crn',
                                                                         common=['crn_log10_A', 'crn_gamma'])])
                                  for psr in psrs])
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = [ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                                             ds.makecommongp_fourier(psrs, ds.powerlaw, components=14, T=T, name='crn',
                                                                     common=['crn_log10_A', 'crn_gamma'])]), p0

        yield ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(components=14), components=30, T=T, name='rednoise',
                                                                    common=['crn_log10_A', 'crn_gamma'])), p0

        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                                 globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw'))
        p0 = ds.sample_uniform(mdl.logL.params)
        yield mdl, p0

        yield ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True),
                                                        ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name='rednoise')]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw')), p0

    def _multipsr_fftcov_likelihoods(self, psrs):
        """Test likelihoods using FFT-based covariance GPs"""
        T = ds.getspan(psrs)
        t0 = ds.getstart(psrs)

        # Test with makecommongp_fftcov for red noise
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fftcov(psrs, ds.powerlaw, components=51, T=T, t0=t0,
                                                                   order=1, name='rednoise'))
        # Access params to initialize and calculate matrices
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with GlobalLikelihood using makeglobalgp_fftcov
        mdl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True),
                                                        ds.makegp_fftcov(psr, ds.powerlaw, components=51, T=T,
                                                                         t0=t0, order=1, name='rednoise')]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf,
                                                                    components=21, T=T, t0=t0, order=1, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with different oversample and fmax_factor parameters
        mdl = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                       ds.makenoise_measurement(psr, psr.noisedict),
                                                       ds.makegp_ecorr(psr, psr.noisedict),
                                                       ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                 commongp = ds.makecommongp_fftcov(psrs, ds.powerlaw, components=51, T=T, t0=t0,
                                                                   order=1, name='rednoise'),
                                 globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf,
                                                                   components=21, T=T, t0=t0, order=1,
                                                                   oversample=5, fmax_factor=2, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

        # Test with monopole ORF
        mdl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                        ds.makenoise_measurement(psr, psr.noisedict),
                                                        ds.makegp_ecorr(psr, psr.noisedict),
                                                        ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                                  globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.monopole_orf,
                                                                    components=21, T=T, t0=t0, order=1, name='gw'))
        params = mdl.logL.params
        p0 = ds.sample_uniform(params)
        yield mdl, p0

    @pytest.mark.integration
    def test_multipsr_fftcov_likelihood(self):
        """Test multi-pulsar likelihoods with FFT-based covariance GPs"""
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile1 = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psrfile2 = data_dir / "v1p1_de440_pint_bipm2019-J0023+0923.feather"
        psrs = [ds.Pulsar.read_feather(psrfile1),
                ds.Pulsar.read_feather(psrfile2)]

        for model in self._multipsr_fftcov_likelihoods(psrs):
            # All models return a suggested parameter set
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            # Verify params are accessible (this initializes matrices)
            assert logl.params is not None
            assert len(logl.params) > 0

    @pytest.mark.integration
    def test_multipsr_likelihood(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile1 = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psrfile2 = data_dir / "v1p1_de440_pint_bipm2019-J0023+0923.feather"
        psrs = [ds.Pulsar.read_feather(psrfile1),
                ds.Pulsar.read_feather(psrfile2)]

        p0_old = None
        for model in self._multipsr_likelihoods(psrs):
            # some models return also a suggested parameter set
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            l1 = logl(p0)
            l2 = jax.jit(logl)(p0)

            assert np.allclose(l1, l2)

            if p0 == p0_old:
                assert np.allclose(l1, l1_old, atol=0.1)

            p0_old, l1_old = p0, l1

    @pytest.mark.integration
    def test_singlepsr_likelihood(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"

        psrfile = data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        psr = ds.Pulsar.read_feather(psrfile)

        p0_old = None
        for model in self._singlepsr_likelihoods(psr):
            if isinstance(model, tuple):
                model, p0 = model
                logl = model.logL
            else:
                logl = model.logL
                p0 = ds.sample_uniform(logl.params)

            l1 = logl(p0)
            l2 = jax.jit(logl)(p0)

            assert np.allclose(l1, l2)

            if p0 and p0 == p0_old:
                assert np.allclose(l1, l1_old, atol=0.1)

            p0_old, l1_old = p0, l1

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
        offset = -52.4

        # Choose the absolute tolerance
        atol = 0.1

        # we need to check the systematic difference between enterprise and discovery
        # before we can run this, but at least we can check the JITted likelihood runs
        # assert float(jax.numpy.abs(ll_difference - offset)) <= atol


@pytest.fixture(scope="module")
def b1855():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    return ds.Pulsar.read_feather(data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather")


def _all_variable_model(psr):
    """Fixed white noise + two variable GP blocks, concat'd into ONE Woodbury
    layer whose N is the bare measurement noise. On the matrix route that
    assembles a `WoodburyKernel_varP` over a `matrix.NoiseMatrix`, which is the
    branch `likelihood.conditional` routes to `make_kernelsolve_simple`."""
    return ds.PulsarLikelihood([psr.residuals,
                                ds.makenoise_measurement(psr, psr.noisedict),
                                ds.makegp_timing(psr, svd=True, variable=True),
                                ds.makegp_fourier(psr, ds.powerlaw, components=10,
                                                  name='rednoise')])


class TestAllVariableConditional:
    """`conditional` on an all-variable single-pulsar model used to crash on the
    matrix route: `WoodburyKernel_varP` had no `make_kernelsolve_simple`.

    Route-agnostic by design — the method is deleted with `matrix.py` when the
    legacy path is removed, but this test keeps running on the surviving path.
    """

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_conditional_returns_both_coefficient_blocks(self, b1855, kernels):
        ds.config(kernels=kernels)
        try:
            model = _all_variable_model(b1855)
            cond = model.conditional
            index = model.N.index

            p0 = ds.sample_uniform(cond.params)
            mu, cf = cond(p0)
        finally:
            ds.config(kernels="matrix")

        # Both blocks are present and keyed by the assembled index names.
        keys = sorted(index)
        assert len(keys) == 2
        assert any('timingmodel_coefficients' in k for k in keys)
        assert any('rednoise_coefficients' in k for k in keys)

        width = sum(sli.stop - sli.start for sli in index.values())
        assert mu.shape == (width,)
        assert np.all(np.isfinite(np.asarray(mu)))

        for par, sli in index.items():
            block = mu[sli]
            assert block.shape == (sli.stop - sli.start,), par

        # cf is the lower-Cholesky factor of Sigma = Pinv + FtNmF.
        L, lower = cf[0], cf[1]
        assert bool(lower) is True
        assert L.shape == (width, width)
        assert np.all(np.diag(np.asarray(L)) > 0.0)

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_sample_conditional_draws_with_the_lower_factor_contract(
            self, b1855, kernels):
        ds.config(kernels=kernels)
        try:
            model = _all_variable_model(b1855)
            sample_cond = model.sample_conditional
            index = model.N.index

            p0 = ds.sample_uniform(sample_cond.params)
            key, c = sample_cond(jax.random.PRNGKey(42), p0)
        finally:
            ds.config(kernels="matrix")

        assert sorted(c) == sorted(index)
        for par, sli in index.items():
            assert c[par].shape == (sli.stop - sli.start,)
            assert np.all(np.isfinite(np.asarray(c[par])))

    def test_matrix_route_ksolve_reports_only_prior_params(self, b1855):
        """The new `make_kernelsolve_simple` reports the P_var params it
        actually reads; N is fixed, so it contributes none."""
        ds.config(kernels="matrix")
        model = _all_variable_model(b1855)
        kernel = model.N

        assert type(kernel).__name__ == "WoodburyKernel_varP"

        ksolve = kernel.make_kernelsolve_simple(model.y)

        assert ksolve.params == sorted(kernel.P_var.make_inv().params)

    def test_conditional_matches_the_certified_metamath_route(self, b1855):
        """The new matrix-route body is proven against the metamath path, which
        the parity suite already certifies. Running is not enough: this pins the
        actual numbers."""
        p0 = None
        out = {}
        for kernels in ("matrix", "metamath"):
            ds.config(kernels=kernels)
            try:
                model = _all_variable_model(b1855)
                cond = model.conditional
                if p0 is None:
                    p0 = ds.sample_uniform(cond.params)
                mu, cf = cond(p0)
                out[kernels] = (np.asarray(mu), np.asarray(cf[0]), bool(cf[1]))
            finally:
                ds.config(kernels="matrix")

        np.testing.assert_allclose(out["matrix"][0], out["metamath"][0], rtol=1e-10)
        # The factor itself is compared more loosely: the two routes assemble
        # Sigma by different (algebraically identical) orderings, so a handful of
        # entries differ at the float64 rounding level.
        np.testing.assert_allclose(out["matrix"][1], out["metamath"][1], rtol=1e-8)
        assert out["matrix"][2] == out["metamath"][2]


class TestConcatShadowGuard:
    """`PulsarLikelihood(concat=False)` chains variable GPs and overwrites
    `.index` each iteration, so every variable GP but the last is silently
    marginalized. That must be confirmed, not stumbled into."""

    def _two_variable_gps(self, psr):
        return [psr.residuals,
                ds.makenoise_measurement(psr, psr.noisedict),
                ds.makegp_fourier(psr, ds.powerlaw, components=10, name='rednoise'),
                ds.makegp_fourier(psr, ds.powerlaw, components=5, name='crn')]

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_unflagged_concat_false_raises_naming_the_shadowed_gps(self, b1855, kernels):
        ds.config(kernels=kernels)
        try:
            with pytest.raises(ValueError, match="shadowed"):
                ds.PulsarLikelihood(self._two_variable_gps(b1855), concat=False)
        finally:
            ds.config(kernels="matrix")

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_the_error_names_the_surviving_and_shadowed_blocks(self, b1855, kernels):
        ds.config(kernels=kernels)
        try:
            with pytest.raises(ValueError) as excinfo:
                ds.PulsarLikelihood(self._two_variable_gps(b1855), concat=False)
        finally:
            ds.config(kernels="matrix")

        message = str(excinfo.value)
        assert "'crn'" in message          # the last GP survives
        assert "rednoise" in message       # the shadowed one is named

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_confirmed_marginalization_constructs(self, b1855, kernels):
        """The flag turns the guard off on both routes."""
        ds.config(kernels=kernels)
        try:
            model = ds.PulsarLikelihood(self._two_variable_gps(b1855), concat=False,
                                        marginalize_all_but_last=True)
        finally:
            ds.config(kernels="matrix")

        assert model.N.index is not None

    def test_confirmed_marginalization_keeps_only_the_last_gp(self, b1855):
        """Scoped to metamath: `clogL` over CHAINED variable GPs is unsupported
        on the legacy matrix route (WoodburyKernel_varNP reaches for
        `make_solve_1d` on its inner WoodburyKernel_varP, which does not define
        it). That gap predates this guard and is out of scope here — the matrix
        path is frozen pending legacy-path removal."""
        ds.config(kernels="metamath")
        try:
            model = ds.PulsarLikelihood(self._two_variable_gps(b1855), concat=False,
                                        marginalize_all_but_last=True)
            params = model.clogL.params
        finally:
            ds.config(kernels="matrix")

        assert any('crn_coefficients' in p for p in params)
        assert not any('rednoise_coefficients' in p for p in params)

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_concat_true_samples_both_coefficient_blocks(self, b1855, kernels):
        ds.config(kernels=kernels)
        try:
            model = ds.PulsarLikelihood(self._two_variable_gps(b1855), concat=True)
            params = model.clogL.params
        finally:
            ds.config(kernels="matrix")

        assert any('crn_coefficients' in p for p in params)
        assert any('rednoise_coefficients' in p for p in params)

    @pytest.mark.parametrize("kernels", ["matrix", "metamath"])
    def test_a_single_variable_gp_is_never_ambiguous(self, b1855, kernels):
        """One variable GP has nothing to shadow: concat=False stays legal."""
        ds.config(kernels=kernels)
        try:
            model = ds.PulsarLikelihood(
                [b1855.residuals,
                 ds.makenoise_measurement(b1855, b1855.noisedict),
                 ds.makegp_fourier(b1855, ds.powerlaw, components=10, name='rednoise')],
                concat=False)
            params = model.clogL.params
        finally:
            ds.config(kernels="matrix")

        assert any('rednoise_coefficients' in p for p in params)


class TestConditionalAllVariable:
    """Regression from main: all-variable GPs must support conditional / sample_conditional.

    Complements :class:`TestAllVariableConditional` with a matrix-route smoke that
    builds ``NoiseMatrix1D_novar`` explicitly (legacy constructor path).
    """

    def _psr(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        return ds.Pulsar.read_feather(
            data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather"
        )

    def test_conditional_variable_timing_plus_red(self):
        ds.config(kernels="matrix")
        try:
            psr = self._psr()
            N = ds.matrix.NoiseMatrix1D_novar(psr.toaerrs ** 2)
            tm = ds.makegp_timing(psr, svd=True, variable=True)
            # name="rednoise" so params match priordict_standard (not bare "red")
            red = ds.makegp_fourier(psr, ds.powerlaw, components=10, name="rednoise")
            like = ds.PulsarLikelihood([psr.residuals, N, tm, red])

            assert type(like.N).__name__ == "WoodburyKernel_varP"
            assert isinstance(like.N.N, ds.matrix.NoiseMatrix)

            p0 = ds.sample_uniform(like.logL.params)
            mu, cf = like.conditional(p0)
            mu = np.asarray(mu)

            n_tm, n_red = tm.F.shape[1], red.F.shape[1]
            assert mu.shape[0] == n_tm + n_red
            names = list(like.N.index)
            assert any("timingmodel_coefficients" in n for n in names)
            assert any("rednoise_coefficients" in n for n in names)

            # Lower-factor contract on the joint precision
            assert cf[1] is True
            L = np.tril(np.asarray(cf[0]))
            assert L.shape == (mu.shape[0], mu.shape[0])

            key = jax.random.PRNGKey(0)
            _, draws = like.sample_conditional(key, p0)
            assert set(draws) == set(like.N.index)
            for name, sli in like.N.index.items():
                assert np.asarray(draws[name]).shape == (sli.stop - sli.start,)
        finally:
            ds.config(kernels="metamath")

    def test_conditional_red_only_does_not_crash(self):
        """Minimal simple-branch smoke: measurement + one variable Fourier GP."""
        ds.config(kernels="matrix")
        try:
            psr = self._psr()
            N = ds.matrix.NoiseMatrix1D_novar(psr.toaerrs ** 2)
            red = ds.makegp_fourier(psr, ds.powerlaw, components=10, name="rednoise")
            like = ds.PulsarLikelihood([psr.residuals, N, red])
            assert type(like.N).__name__ == "WoodburyKernel_varP"

            p0 = ds.sample_uniform(like.logL.params)
            mu, cf = like.conditional(p0)
            assert np.asarray(mu).shape[0] == red.F.shape[1]
            assert cf[1] is True
        finally:
            ds.config(kernels="metamath")
