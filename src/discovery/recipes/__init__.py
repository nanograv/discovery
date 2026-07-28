"""Ready-made model-building recipes (a small "model zoo").

Import it explicitly (it is not pulled into the top-level ``discovery``
namespace). Each function assembles a Discovery likelihood from the public API
and returns it, e.g.::

    import discovery as ds
    import discovery.recipes as ds_recipes

    model = ds_recipes.full_rn(psr)
    logL  = model.logL(ds.sample_uniform(model.logL.params))

This module is also the **single source of truth** for the parity test-suite
(``tests/metamatrix/test_{pulsar,global,array}.py``) and the docs cookbook
(``docs/tutorials/cookbook_models.ipynb``) — both import these same builders, so
every recipe is exercised by a test that asserts the ``matrix`` and ``metamath``
kernel backends agree. Every recipe works unchanged under either backend
(``ds.config(kernels='matrix'|'metamath')``).

Each function's one-line docstring is the cookbook caption; keep it to a single
sentence. The ordered ``SINGLE_PULSAR`` / ``GLOBAL`` / ``ARRAY`` lists drive the
cookbook's table of contents.
"""
import numpy as np
import jax.numpy as jnp

import discovery as ds


# ---------------------------------------------------------------------------
# Single-pulsar recipes — PulsarLikelihood([...])
# ---------------------------------------------------------------------------

def measurement_simple(psr):
    """White noise only, single-backend (efac + t2equad), no selection."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement_simple(psr),
    ])


def measurement_white(psr):
    """White noise with per-backend efac/equad selection."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr),
    ])


def ecorr_gp(psr):
    """White noise plus ECORR modelled as a separate Gaussian-process component."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr),
        ds.makegp_ecorr(psr),
    ])


def ecorr_sm(psr):
    """ECORR folded into the noise matrix via Sherman-Morrison (+ timing GP wrapper)."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict, ecorr=True),
        ds.makegp_timing(psr, svd=True),
    ])


def meas_timing(psr):
    """White noise plus an (SVD-stabilised) marginalised timing-model GP."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
    ])


def full_rn(psr):
    """Realistic single-pulsar model: white + ECORR-GP + timing + power-law red noise."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def full_rn_concat_false(psr):
    """Same as full_rn but with concat=False → chained (nested) Woodbury kernels."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ], concat=False)


def multi_vgp(psr):
    """Two variable GPs: achromatic red noise + a chromatic DM GP (on the DM Fourier basis)."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
        # DM noise is chromatic -> use the DM (nu^-2) Fourier basis, not the default
        ds.makegp_fourier(psr, ds.powerlaw, components=14, name="dmgp",
                          fourierbasis=ds.dmfourierbasis),
    ])


def variable_timing(psr):
    """Timing model as a *variable* GP (coefficients sampled, not marginalised) + RN."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, ecorr=True),
        ds.makegp_timing(psr, svd=True, variable=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def fftcov_2d(psr):
    """Red noise via an FFT-derived dense (2D) covariance basis (makegp_fftcov)."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fftcov(psr, ds.powerlaw, components=31, name="rednoise"),
    ])


def _toy_delay(toas):
    # deterministic, parameter-free delay (args come only from psr attributes).
    return 1e-9 * jnp.sin(2.0 * jnp.pi * (toas - toas.min()) / 3.16e8)


def delay(psr):
    """A deterministic delay subtracted from the residuals (makedelay → CompoundDelay)."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makedelay(psr, _toy_delay, name="toydelay"),
    ])


def fourier_variance_fixed(psr):
    """A Fourier GP whose prior covariance is supplied directly as a fixed matrix."""
    comps = 10
    argname = f"{psr.name}_fourierGP_variance({comps * 2},{comps * 2})"
    cov = np.diag(np.full(comps * 2, 1e-4))
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier_variance(psr, components=comps, noisedict={argname: cov}),
    ])


# ---------------------------------------------------------------------------
# Multi-pulsar: GlobalLikelihood — per-pulsar models + optional correlated GP
# ---------------------------------------------------------------------------

def _psl_with_rn(psr, T):
    # per-pulsar model carrying its own red noise (for GlobalLikelihood rows)
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name="rednoise"),
    ])


def no_global(psrs):
    """Independent pulsars — GlobalLikelihood with no correlated GP (sum of per-psr logL)."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood([_psl_with_rn(p, T) for p in psrs])


def global_hd(psrs):
    """A Hellings-Downs-correlated common GW signal across pulsars (dense 2D prior)."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood(
        [_psl_with_rn(p, T) for p in psrs],
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
    )


def global_monopole(psrs):
    """A monopole-correlated common signal across pulsars."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood(
        [_psl_with_rn(p, T) for p in psrs],
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.monopole_orf,
                                         components=14, T=T, name="gw"),
    )


def global_compound(psrs):
    """Two correlated global GPs at once (HD + monopole) via a globalgp list (CompoundGlobalGP)."""
    T = ds.getspan(psrs)
    return ds.GlobalLikelihood(
        [_psl_with_rn(p, T) for p in psrs],
        globalgp=[
            ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                    components=14, T=T, name="gw"),
            ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.monopole_orf,
                                    components=14, T=T, name="gw_mono"),
        ],
    )


# ---------------------------------------------------------------------------
# Multi-pulsar: ArrayLikelihood — vectorised, with commongp / globalgp / extsignals
# ---------------------------------------------------------------------------

def _psl_skeleton(psr):
    # per-pulsar model WITHOUT red noise (red noise lives in the commongp)
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
    ])


def no_common(psrs):
    """ArrayLikelihood with per-pulsar red noise inline, no shared/correlated GP."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood([
        ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name="rednoise"),
        ]) for psr in psrs
    ])


def intrinsic_rn(psrs):
    """Per-pulsar intrinsic red noise on a shared Fourier basis (vectorised; independent amplitudes per pulsar)."""
    # NOTE: a `commongp` with no `common=[...]` params is *intrinsic* (per-pulsar)
    # red noise, vectorised over the array — NOT a common/correlated process.
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
    )


def intrinsic_plus_crn(psrs):
    """Per-pulsar intrinsic red noise + a common-spectrum (CRN) process on one shared basis, via make_combined_crn."""
    # The idiomatic way to put intrinsic RN and a common-spectrum process on the
    # same basis: build a single combined PSD; CRN params (gw_*) are shared
    # across pulsars (passed as `common`), intrinsic params stay per-pulsar.
    T = ds.getspan(psrs)
    combined, crn_params = ds.make_combined_crn(14, ds.powerlaw, ds.powerlaw,
                                                crn_prefix="gw_")
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, combined, components=30,
                                         T=T, name="rednoise", common=crn_params),
    )


def intrinsic_rn_plus_global_hd(psrs):
    """Per-pulsar intrinsic red noise plus an HD-correlated global GW signal (the canonical PTA model)."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
    )


def decenter_intrinsic_rn(psrs):
    """intrinsic_rn built in a decentered (whitened-coefficient) parameterisation."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        decenter=True,
    )


def decenter_intrinsic_rn_global_hd(psrs):
    """Decentered intrinsic red noise + HD global GP (decentered sampling of the full model)."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"),
        decenter=True,
    )


def means_on_commongp(psrs):
    """An (intrinsic-RN) common GP with a non-zero prior mean supplied by a `means` callable."""
    def my_means(f, df, mean_amp):
        return mean_amp * jnp.ones_like(f)

    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise", means=my_means),
    )


def extsignal_cw(psrs):
    """Intrinsic red noise plus a continuous-wave deterministic signal on its own basis."""
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl_skeleton(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30,
                                         T=T, name="rednoise"),
        extsignals=[
            ds.makecw_extsignal(psrs, components=50, T=T, pulsarterm=True, name="cw"),
        ],
    )


# ---------------------------------------------------------------------------
# Ordered catalogs (drive the cookbook TOC; tests build their tables from these)
# ---------------------------------------------------------------------------

SINGLE_PULSAR = [
    measurement_simple, measurement_white, ecorr_gp, ecorr_sm, meas_timing,
    full_rn, full_rn_concat_false, multi_vgp, variable_timing,
    fftcov_2d, delay, fourier_variance_fixed,
]

GLOBAL = [no_global, global_hd, global_monopole, global_compound]

ARRAY = [
    no_common, intrinsic_rn, intrinsic_plus_crn, intrinsic_rn_plus_global_hd,
    decenter_intrinsic_rn, decenter_intrinsic_rn_global_hd, means_on_commongp, extsignal_cw,
]
