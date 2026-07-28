"""Graph-precision pass, stage 1 -- blanket working dtype at materialization.

The metamath materialization boundary (metamatrix.build_callable_from_graph,
reached via func()) casts every floating leaf -- args, constants, PSD/func-leaf
outputs -- to utils.working_dtype() when single precision is active. Stage 1 is
the *blanket, no-pins* baseline: the ENTIRE Woodbury (including the would-be f64
pins ytNmy / logdets) runs in the working dtype. These tests lock that path:

  * float64 default is untouched (the cast is the identity unless working=f32);
    covered structurally by the metamatrix parity suite, asserted here too.
  * under working=float32 the single-pulsar logL is finite and close to f64
    (no pins are needed at single-pulsar scale -- the ytNmy cancellation only
    bites across the full array; that motivates the pins, added next).

Requires the metamath backend (ds.config(kernels='metamath')) -- the cast lives
in the metamath graph; the legacy matrix.py path is unaffected.
"""
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import utils


DATA = Path(__file__).resolve().parents[2] / "data"
B1855 = DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"

# Three long-baseline pulsars for the array/fused-path test.
ARRAY_FEATHERS = [
    "v1p1_de440_pint_bipm2019-B1855+09.feather",
    "v1p1_de440_pint_bipm2019-B1937+21.feather",
    "v1p1_de440_pint_bipm2019-J1909-3744.feather",
]


@pytest.fixture(scope="module")
def psr():
    return ds.Pulsar.read_feather(B1855)


@pytest.fixture(scope="module")
def psrs():
    return [ds.Pulsar.read_feather(DATA / f) for f in ARRAY_FEATHERS]


@contextlib.contextmanager
def metamath_working(dtype):
    """Metamath backend + a chosen working dtype; restore matrix/float64 after.
    Build the model INSIDE the block so its graph materializes under both."""
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


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

# Whole single-pulsar Woodbury in f32 (no pins): logL holds to ~1e-8 relative.
BLANKET_RTOL = 1e-7


def _draw(model, seed=0):
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


@pytest.mark.parametrize("build", BUILDERS)
def test_metamath_blanket_f32(psr, build):
    # float64 reference, metamath backend
    with metamath_working(jnp.float64):
        ref = build(psr)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    assert np.isfinite(L64)

    # blanket float32: the entire graph in working dtype
    with metamath_working(jnp.float32):
        m = build(psr)
        L32 = float(m.logL(p0))

    assert np.isfinite(L32), f"{build.__name__} blanket-f32 logL not finite"
    np.testing.assert_allclose(
        L32, L64, rtol=BLANKET_RTOL,
        err_msg=f"{build.__name__} blanket-f32 logL {L32} vs f64 {L64}")


def test_cast_is_active_in_metamath_f32(psr):
    """Guard against silent no-op: blanket f32 must actually move logL off the
    f64 value (the whole graph really is running in f32, not just Phi)."""
    with metamath_working(jnp.float64):
        ref = build_full_rn(psr)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    with metamath_working(jnp.float32):
        L32 = float(build_full_rn(psr).logL(p0))
    assert abs(L64 - L32) > 0.0, "blanket f32 produced bit-identical logL -- cast inert?"


def test_float64_default_unchanged(psr):
    """The cast must be the identity in the default (float64) regime."""
    assert utils.working_dtype() is jnp.float64 and not utils.single_precision
    with metamath_working(jnp.float64):
        m = build_full_rn(psr)
        p0 = _draw(m)
        L = float(m.logL(p0))
    assert np.isfinite(L)


# --- Stage 2a: dtype-map + cast-on-read machinery (pins not wired in yet) ------
#
# 2a builds the per-edge precision mechanism with the pin set EMPTY. With no
# pins it must behave exactly like stage 1 (blanket working dtype) -- the model
# tests above already lock that. These unit tests lock the new helper directly.

from collections import OrderedDict

from discovery import metamatrix as mm


def _toy_graph():
    """a, b leaves -> c = a @ b -> d = c.sum(). Returns (graph, names)."""
    g = mm.GraphBuilder()
    a = g.leaf(None, name="a")
    b = g.leaf(None, name="b")
    c = g.node(lambda x, y: x @ y, [a, b], name="c")
    d = g.node(lambda x: x.sum(), [c], name="d")
    return g.graph, ("a", "b", "c", "d")


def test_dtype_map_empty_pins_all_working():
    """No pins -> every node maps to the working dtype (float32 in single mode)."""
    graph, names = _toy_graph()
    with metamath_working(jnp.float32):
        dm = mm._dtype_map(graph, jnp.float32)
    assert all(dm[n] is jnp.float32 for n in names)


def test_dtype_map_float64_regime_is_identity():
    """In the float64 default regime the map is all float64 (cast is a no-op)."""
    graph, names = _toy_graph()
    dm = mm._dtype_map(graph, jnp.float64)
    assert all(dm[n] is jnp.float64 for n in names)


def test_dtype_map_pin_marks_node_and_ancestors():
    """Pinning a node forces it and everything it depends on to float64; nodes
    that do not feed the pin stay working dtype."""
    g = mm.GraphBuilder()
    a = g.leaf(None, name="a")          # feeds pinned d
    b = g.leaf(None, name="b")          # feeds pinned d
    e = g.leaf(None, name="e")          # feeds only unpinned f
    d = g.node(lambda x, y: x @ y, [a, b], name="d")
    f = g.node(lambda x: x.sum(), [e], name="f")
    g.pin_f64(d)

    with metamath_working(jnp.float32):
        dm = mm._dtype_map(g.graph, jnp.float32)

    assert dm["d"] is jnp.float64                 # the pin
    assert dm["a"] is jnp.float64 and dm["b"] is jnp.float64  # its ancestors
    assert dm["e"] is jnp.float32 and dm["f"] is jnp.float32  # off the pin's path


def test_dtype_map_combine_f64_does_not_propagate():
    """combine_f64 (Half A) marks ONLY its own node float64; its ancestors stay
    working dtype -- the opposite of pin_f64, which propagates float64 backward.
    This is what lets the final logL combination run in float64 while the
    expensive upstream factorization stays float32."""
    g = mm.GraphBuilder()
    a = g.leaf(None, name="a")
    b = g.leaf(None, name="b")
    c = g.node(lambda x, y: x @ y, [a, b], name="c")   # an f32 ancestor of the combine
    g.combine_f64(g.node(lambda x: -0.5 * x, [c], name="d"))

    with metamath_working(jnp.float32):
        dm = mm._dtype_map(g.graph, jnp.float32)

    assert dm["d"] is jnp.float64                 # the combine node computes in f64
    assert dm["c"] is jnp.float32                 # ancestor NOT pulled to f64 (vs pin)
    assert dm["a"] is jnp.float32 and dm["b"] is jnp.float32


# --- Stage 2b: the woodbury pins (ytNmy, lN, lP) on a real model graph --------
#
# woodbury pins y^T N^-1 y and the white-noise / prior logdets to float64. Under
# variable white noise these stay *live* (they don't fold to constants), so we
# can read the materialized graph and check the dtype the per-edge cast assigns
# each node. The small Phi^-1 + FtNmF Cholesky (and its logdet lS) must stay in
# the working dtype -- pinning it would force an expensive float64 Cholesky.


def build_live_wn(psr):
    """Variable white noise (efac/equad sampled) -> N, ytNmy, FtNmF all live."""
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr),          # no noisedict -> sampled WN
        ds.makegp_timing(psr, svd=True),
        ds.makegp_fourier(psr, ds.powerlaw, components=30, name="rednoise"),
    ])


def _walk_dtypes(graph, working):
    """Yield (qualified_name, node, dtype) for every Node in graph and its
    subgraphs, computing the dtype map per (sub)graph exactly as
    build_callable_from_graph does (it recurses into each GraphLeaf)."""
    from discovery import metamatrix as mm
    dm = mm._dtype_map(graph, working)
    for name, node in graph.items():
        if isinstance(node, mm.GraphLeaf):
            yield from _walk_dtypes(node.graph, working)
        elif isinstance(node, mm.Node):
            yield name, node, dm[name]


def test_woodbury_pins_are_f64_cholesky_is_f32(psr):
    """On a live-WN model graph: every pinned node (ytNmy, lN, lP) is float64,
    and every cho_factor node (the lS Cholesky) is the working float32."""
    from discovery import metamatrix as mm

    mm.keepgraph = True
    try:
        with metamath_working(jnp.float32):
            m = build_live_wn(psr)
            # Walk the dtype map *inside* the float32 context: _dtype_map reads
            # the active single-precision config, so it must be computed here.
            walked = list(_walk_dtypes(m.logL.graph, jnp.float32))
    finally:
        mm.keepgraph = False

    pins = [(n, d) for n, node, d in walked if getattr(node, "pin", False)]
    chos = [(n, d) for n, node, d in walked
            if node.description and "cho_factor" in node.description]

    assert pins, "no pinned nodes survived in the live-WN graph"
    assert chos, "no cho_factor node found -- test is vacuous"
    assert all(d is jnp.float64 for _, d in pins), \
        f"a pinned node is not float64: {pins}"
    assert all(d is jnp.float32 for _, d in chos), \
        f"a cho_factor (lS Cholesky) is not the working float32: {chos}"


def test_live_wn_logL_finite_and_close_with_pins(psr):
    """Variable-WN logL under float32 (with pins) is finite, close to float64,
    and has finite gradients (sampler safety)."""
    with metamath_working(jnp.float64):
        ref = build_live_wn(psr)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    assert np.isfinite(L64)

    with metamath_working(jnp.float32):
        m = build_live_wn(psr)
        L32 = float(m.logL(p0))
        grads = jax.grad(lambda p: m.logL(p))(p0)

    assert np.isfinite(L32), "live-WN float32 logL not finite"
    np.testing.assert_allclose(L32, L64, rtol=1e-5,
                               err_msg=f"live-WN float32 logL {L32} vs f64 {L64}")
    assert all(np.isfinite(np.asarray(v)).all() for v in grads.values()), \
        "live-WN float32 gradient not finite"


def test_pin_forces_f64_accumulation():
    """End-to-end proof that a pin actually changes the computed result, not
    just the dtype map. A dot product with catastrophic cancellation in its sum
    (1e8 + 1 - 1e8 = 1) loses everything in float32 but is recovered when the
    node is pinned (the sum accumulates in float64, and 1.0 survives the downcast
    to the float32 consumer). This is the load-bearing behaviour the whole stage
    exists for; if the cast ever silently stops happening, this flips to 0.0."""
    from discovery import metamatrix as mm

    y = np.array([1e8, 1.0, -1e8])
    Nmy = np.array([1.0, 1.0, 1.0])

    def make(pin):
        g = mm.GraphBuilder()
        a = g.leaf(None, name="y")
        b = g.leaf(None, name="Nmy")
        d = g.dot(a, b)             # y^T Nmy -- the (optionally) pinned op
        if pin:
            g.pin_f64(d)
        _ = 0.0 + d                 # a downstream working-dtype (f32) consumer
        return mm.func(g.graph)

    ds.config(kernels="metamath")
    utils.config(backend="jax", working=jnp.float32)
    try:
        assert float(make(True)(y, Nmy)) == 1.0,  "pinned dot lost f64 accumulation"
        assert float(make(False)(y, Nmy)) == 0.0, "unpinned f32 dot unexpectedly survived"
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


# --- Stage 2c: the array / fused likelihood path in float32 --------------------
#
# THIS is the coverage gap the array+fused pins were added to protect. Everything
# above runs either the single-pulsar woodbury or the dtype-map helper in
# isolation. An ArrayLikelihood with BOTH a commongp (shared Fourier basis) and a
# globalgp (correlated HD GP, Phi a NoiseMatrix) routes through
# GlobalWoodburyKernel.make_kernelproduct -> the fused branch
# (vectorwoodburyjointsolve + globalwoodbury_fused) -- exactly where this
# session's new pins live and had never been exercised in float32.
#
# Two things are checked: (1) the f32 logL is finite, close to f64, and has finite
# gradients (sampler safety -- the real array data point); (2) the dtype-map
# structure on the fused graph is correct: every pinned node (raw per-pulsar
# y^T N^-1 y, lN, lP) is float64, and every cho_factor (inner per-pulsar + outer
# global Cholesky, the lS terms) stays the working float32. The white-box check is
# the real payoff: it locks the "pin the raw y^T N^-1 y at source, NOT the
# projected ytNmy_proj" decision -- pinning the projected term would silently drag
# the inner Cholesky to f64 and kill the f32 speed win.


def _array_fused(psrs, fixed_wn):
    """ArrayLikelihood hitting the fused path (commongp + globalgp).

    fixed_wn=True  -> white noise pinned to the realistic noisedict: a well-
                      conditioned N, robust across prior draws. The f64 pins fold
                      to constants here (still computed in f64 at fold time), so
                      they do their job but are invisible in the dtype map.
    fixed_wn=False -> sampled white noise: keeps the pins LIVE so they appear in
                      the dtype map. NOTE: sample_uniform draws physically extreme
                      WN (efac up to ~10) that can badly scale N and overflow the
                      f32 Cholesky -- so this variant is only used for the dtype
                      structure check, not for a logL-closeness assertion."""
    T = ds.getspan(psrs)
    def noise(psr):
        return (ds.makenoise_measurement(psr, psr.noisedict) if fixed_wn
                else ds.makenoise_measurement(psr))
    return ds.ArrayLikelihood(
        [ds.PulsarLikelihood([psr.residuals,
                              noise(psr),
                              ds.makegp_ecorr(psr, psr.noisedict),
                              ds.makegp_timing(psr, svd=True)]) for psr in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="rednoise"),
        globalgp=ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                         components=14, T=T, name="gw"))


def test_array_fused_logL_finite_and_close(psrs):
    """3-pulsar fused-path logL under float32 is finite, close to float64, and has
    finite gradients. First real data point on the array ytNmy cancellation -- at
    3-pulsar scale (with realistic fixed white noise) it holds easily; the pins are
    insurance for the full-array regime. This proves the fused f32 path runs and is
    sane. (Sampled-WN draws can overflow the f32 Cholesky on physically extreme
    efac/equad points -- a conditioning artifact of sample_uniform's broad priors,
    not a fused-path precision failure; see _array_fused's note.)"""
    with metamath_working(jnp.float64):
        ref = _array_fused(psrs, fixed_wn=True)
        p0 = _draw(ref)
        L64 = float(ref.logL(p0))
    assert np.isfinite(L64)

    with metamath_working(jnp.float32):
        m = _array_fused(psrs, fixed_wn=True)
        L32 = float(m.logL(p0))
        grads = jax.grad(lambda p: m.logL(p))(p0)

    assert np.isfinite(L32), "array fused-path float32 logL not finite"
    np.testing.assert_allclose(L32, L64, rtol=1e-5,
                               err_msg=f"array fused float32 logL {L32} vs f64 {L64}")
    assert all(np.isfinite(np.asarray(v)).all() for v in grads.values()), \
        "array fused-path float32 gradient not finite"


def test_array_fused_pins_are_f64_cholesky_is_f32(psrs):
    """On the fused-path graph (incl. the nested vectorwoodburyjointsolve subgraph,
    which _walk_dtypes recurses into): every pinned node is float64, every
    cho_factor (inner per-pulsar + outer global Cholesky) is the working float32.
    Sampled WN (fixed_wn=False) keeps the pins live so they show in the map."""
    from discovery import metamatrix as mm

    mm.keepgraph = True
    try:
        with metamath_working(jnp.float32):
            m = _array_fused(psrs, fixed_wn=False)
            walked = list(_walk_dtypes(m.logL.graph, jnp.float32))
    finally:
        mm.keepgraph = False

    pins = [(n, d) for n, node, d in walked if getattr(node, "pin", False)]
    chos = [(n, d) for n, node, d in walked
            if node.description and "cho_factor" in node.description]

    assert pins, "no pinned nodes survived in the fused-path graph"
    assert chos, "no cho_factor node found -- fused path not exercised?"
    assert all(d is jnp.float64 for _, d in pins), \
        f"a pinned node is not float64: {pins}"
    assert all(d is jnp.float32 for _, d in chos), \
        f"a cho_factor (lS Cholesky) is not the working float32: {chos}"
