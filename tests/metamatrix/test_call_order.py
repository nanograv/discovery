"""Call-order invariance of `ArrayLikelihood`'s cached properties.

`logL` / `clogL` / `conditional` used to each rebuild the kernel assembly and
write it to `self.vsm` / `self.ys` as a side effect, with variations. Whichever
property you touched first therefore decided what the others saw. These tests
pin the invariant the two cached assembly helpers restore: every property agrees
with a fresh single-purpose instance, whatever order it is built in.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402


RTOL = 1e-12


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


def _psl(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
    ])


def _commongp_only(psrs):
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T,
                                         name="rednoise"))


def _commongp_globalgp_decenter(psrs):
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl(p) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T,
                                         name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=5, T=T, name="gw"),
        decenter=True)


def _clogl(model, p0):
    """`clogL` prunes to the 'staged' (logp, c) pair when reparams are present
    (e.g. decenter=True) and to the bare 'logp' scalar otherwise."""
    out = model.clogL(p0)
    return float(out[0]) if isinstance(out, tuple) else float(out)


def _params(model, rng):
    """Hyperparameters from the standard priors, coefficients drawn directly
    (coefficient keys have no entry in the prior dictionary)."""
    keys = sorted(set(model.logL.params) | set(model.clogL.params))
    coeff = [k for k in keys if "_coefficients(" in k]
    p0 = ds.sample_uniform([k for k in keys if k not in coeff])
    for k in coeff:
        width = int(k[k.index("(") + 1:k.index(")")])
        p0[k] = rng.normal(size=width)
    return p0


def test_commongp_globalgp_decenter_is_call_order_invariant(psrs, metamath_backend):
    rng = np.random.default_rng(20260716)
    p0 = _params(_commongp_globalgp_decenter(psrs), rng)

    # fresh single-purpose instances: the reference values
    ref_logl = float(_commongp_globalgp_decenter(psrs).logL(p0))
    ref_clogl = _clogl(_commongp_globalgp_decenter(psrs), p0)

    # logL first, then clogL
    a = _commongp_globalgp_decenter(psrs)
    a_logl, a_clogl = float(a.logL(p0)), _clogl(a, p0)

    # clogL first, then logL
    b = _commongp_globalgp_decenter(psrs)
    b_clogl, b_logl = _clogl(b, p0), float(b.logL(p0))

    np.testing.assert_allclose(a_logl, ref_logl, rtol=RTOL)
    np.testing.assert_allclose(b_logl, ref_logl, rtol=RTOL)
    np.testing.assert_allclose(a_clogl, ref_clogl, rtol=RTOL)
    np.testing.assert_allclose(b_clogl, ref_clogl, rtol=RTOL)


def test_commongp_only_conditional_first_then_the_rest(psrs, metamath_backend):
    """`conditional` is the property that used to short-circuit on an already-set
    `self.vsm`, so it is the one most sensitive to order. It supports commongp
    only -- this test never asks it to handle a global GP."""
    rng = np.random.default_rng(20260716)
    p0 = _params(_commongp_only(psrs), rng)

    ref_logl = float(_commongp_only(psrs).logL(p0))
    ref_clogl = _clogl(_commongp_only(psrs), p0)
    ref_mu, ref_cf = _commongp_only(psrs).conditional(p0)

    # conditional first
    a = _commongp_only(psrs)
    a_mu, _ = a.conditional(p0)
    a_logl, a_clogl = float(a.logL(p0)), _clogl(a, p0)

    # conditional last
    b = _commongp_only(psrs)
    b_clogl, b_logl = _clogl(b, p0), float(b.logL(p0))
    b_mu, _ = b.conditional(p0)

    np.testing.assert_allclose(a_logl, ref_logl, rtol=RTOL)
    np.testing.assert_allclose(b_logl, ref_logl, rtol=RTOL)
    np.testing.assert_allclose(a_clogl, ref_clogl, rtol=RTOL)
    np.testing.assert_allclose(b_clogl, ref_clogl, rtol=RTOL)
    np.testing.assert_allclose(np.asarray(a_mu), np.asarray(ref_mu), rtol=RTOL)
    np.testing.assert_allclose(np.asarray(b_mu), np.asarray(ref_mu), rtol=RTOL)


def test_sample_conditional_index_is_not_clobbered_by_clogl(psrs, metamath_backend):
    """`sample_conditional` reads the assembly's index. Building `clogL` first
    used to overwrite `self.vsm` with the coefficient assembly, whose index
    differs whenever a globalgp is folded in."""
    a = _commongp_only(psrs)
    ref_index = dict(a._marginal_assembly[0].index)

    b = _commongp_only(psrs)
    b.clogL                                   # noqa: B018 — force the other assembly
    after_index = dict(b._marginal_assembly[0].index)

    assert after_index == ref_index


def test_the_two_assemblies_are_distinct_objects_and_both_cached(psrs, metamath_backend):
    model = _commongp_globalgp_decenter(psrs)

    marg, coeff = model._marginal_assembly, model._coefficient_assembly

    assert marg[0] is not coeff[0]
    # caching: the same tuple comes back, no rebuild
    assert model._marginal_assembly is marg
    assert model._coefficient_assembly is coeff
    # the coefficient assembly folds the globalgp in; the marginal one does not
    assert isinstance(coeff[0].index, list)
    assert set(marg[0].index) != {k for d in coeff[0].index for k in d}


def test_reference_is_consulted_only_by_the_marginal_assembly(psrs, metamath_backend):
    """reference+delta affects the marginal paths only; the coefficient
    assembly deliberately ignores it."""
    T = ds.getspan(psrs)

    def build(reference):
        return ds.ArrayLikelihood(
            [_psl(p) for p in psrs],
            commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=10,
                                             T=T, name="rednoise"),
            reference=reference)

    plain = build(None)
    theta_ref = ds.sample_uniform(plain.logL.params)
    model = build(theta_ref)

    assert hasattr(model._marginal_assembly[0], "P_ref")
    assert not hasattr(model._coefficient_assembly[0], "P_ref")
