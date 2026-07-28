"""Phase D -- single precision (mixed), CPU.

x64 stays ON; we flip the working dtype to float32 via utils.config(working=...).
The PSD factories now cast their output Phi to working_dtype() (Phase A/the cast
added at the signals boundary), so under this knob Phi is genuinely float32.

What this regime is -- and isn't:
  * The model's F / N / FtNmF stay float64; JAX re-promotes the float32 Phi to
    float64 the moment they meet (the plan's D2 caveat). So Piece 1 is
    *NaN-safety*, not an end-to-end float32 speedup -- the float32 -> fast path
    is Piece 2 (the graph backward pass).
  * NaN-safety here is structural: the clip pins Phi to [1e-18, 1e-9] s^2, both
    comfortably inside float32's normal range (~1.2e-38 .. 3.4e38), so the cast
    can never produce inf/denormal/zero -- even at pathological prior edges where
    a steep f^-gamma tail would otherwise underflow.

Tests:
  D1 finiteness -- logL finite under float32, including at steep prior edges.
  D2 accuracy   -- |logL_f32 - logL_f64| within a float32-scale tolerance.
  D3 gradients  -- jax.grad(logL) finite under float32 (sampler safety).
"""
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)   # required for the float32 *working* knob

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import utils


DATA = Path(__file__).resolve().parents[2] / "data"
B1855 = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"


@pytest.fixture(scope="module")
def psr():
    return ds.Pulsar.read_feather(B1855)


@contextlib.contextmanager
def working(dtype):
    """Switch the working dtype, restoring float64 afterward. Build the model
    *inside* the block so its graph traces under the active dtype."""
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")


def build_full_rn(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def build_multi_vgp(psr):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
        ds.makegp_fourier(psr, ds.powerlaw, components=14, name="dmgp",
                          fourierbasis=ds.dmfourierbasis),
    ])


BUILDERS = [pytest.param(build_full_rn, id="full_rn"),
            pytest.param(build_multi_vgp, id="multi_vgp")]

# Float32 only carries ~7 significant digits; the Phi rounds to ~6e-8 relative,
# which propagates to logL. 1e-4 relative is the float32-scale tolerance the
# plan calls for -- loose enough for f32, tight enough to catch a real break.
F32_RTOL = 1e-4


def _fixed_params(model, seed=0):
    """Draw a clip-inert parameter point (float64 reference regime)."""
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


def _edge_params(model):
    """A steep prior edge: max gamma so the f^-gamma tail dives below the floor
    (exercises the clip's NaN-safety in float32)."""
    p = {}
    for name in model.logL.params:
        if name.endswith("_gamma"):
            p[name] = 6.99
        elif name.endswith("_log10_A"):
            p[name] = -12.5
        else:
            np.random.seed(abs(hash(name)) % (2 ** 31))
            p[name] = float(ds.sample_uniform([name])[name])
    return p


# --- D1: finiteness -----------------------------------------------------------
@pytest.mark.parametrize("build", BUILDERS)
def test_d1_finite(psr, build):
    ref = build(psr)
    p0 = _fixed_params(ref)
    p_edge = _edge_params(ref)
    with working(jnp.float32):
        m = build(psr)
        for p in (p0, p_edge):
            val = float(m.logL(p))
            assert np.isfinite(val), f"{build.__name__} logL not finite: {val}"


# --- D2: accuracy vs float64 --------------------------------------------------
@pytest.mark.parametrize("build", BUILDERS)
def test_d2_accuracy(psr, build):
    ref = build(psr)
    p0 = _fixed_params(ref)
    L64 = float(ref.logL(p0))
    with working(jnp.float32):
        m = build(psr)
        L32 = float(m.logL(p0))
    assert np.isfinite(L32)
    np.testing.assert_allclose(
        L32, L64, rtol=F32_RTOL,
        err_msg=f"{build.__name__} float32 logL {L32} vs float64 {L64}")


# --- D3: gradients ------------------------------------------------------------
@pytest.mark.parametrize("build", BUILDERS)
def test_d3_grad_finite(psr, build):
    ref = build(psr)
    p0 = _fixed_params(ref)
    names = list(p0)
    x0 = jnp.array([p0[n] for n in names], dtype=jnp.float32)

    with working(jnp.float32):
        m = build(psr)

        def logL_vec(x):
            return m.logL({n: x[i] for i, n in enumerate(names)})

        g = jax.grad(logL_vec)(x0)
    g = np.asarray(g)
    assert g.shape == (len(names),)
    assert np.all(np.isfinite(g)), (
        f"{build.__name__} non-finite grad at "
        f"{[names[i] for i in np.where(~np.isfinite(g))[0]]}")
