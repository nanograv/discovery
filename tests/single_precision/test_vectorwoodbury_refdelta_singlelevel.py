"""Test rung for the single-level, batched reference+delta graph
``mh.vectorwoodbury_refdelta`` -- the production form for no-Hellings-Downs array
analyses (per-pulsar IRN, CURN, IRN+CURN via make_combined_crn), where each pulsar
has one diagonal-Phi red-noise GP and the array logL factorises (logL = sum_i logL_i).

It is the batched twin of the scalar ``woodbury_refdelta``:

  * equals the ordinary ``vectorwoodbury`` logL exactly in float64, any Phi/Phi_ref;
  * is independent of the reference Phi_ref (pure reparametrisation);
  * in float32 holds logL far tighter than direct ``vectorwoodbury`` (no
    sum_i(ytNmy - FtNmy.mu) cancellation of ~1e6 numbers across many pulsars);
  * keeps the batched GP-block Cholesky in float32 (the speed win).

See research_note_on_split_with_reference.md sec. 2-3, docs/adr/0001,0003, and
docs/metamatrix_dev.md (fused reference+delta).
"""
import contextlib

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from discovery import metamath as mh
from discovery import metamatrix as mm
from discovery import utils

NP, K = 8, 14
NTOA = [400 + 23 * i for i in range(NP)]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    ys = [jnp.asarray(25.0 * rng.standard_normal(n)) for n in NTOA]
    Ns = [mh.NoiseMatrix(jnp.asarray(rng.uniform(0.5, 2.0, n))) for n in NTOA]
    Fs = [jnp.asarray(rng.standard_normal((n, K))) for n in NTOA]
    phi_base = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, K)))
    phi_ref = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, K)))
    return dict(ys=ys, Ns=Ns, Fs=Fs, phi_base=phi_base, phi_ref=phi_ref)


def _solves(m):
    return [N.make_solve for N in m["Ns"]]


def _vw(m, phi):
    P = mh.NoiseMatrix(phi)
    return float(mm.func(mh.vectorwoodbury(m["ys"], _solves(m), m["Fs"], P.make_inv))(params={}))


def _rd(m, phi, phi_ref):
    P, Pr = mh.NoiseMatrix(phi), mh.NoiseMatrix(phi_ref)
    return float(mm.func(mh.vectorwoodbury_refdelta(
        m["ys"], _solves(m), m["Fs"], P.make_inv, Pr.make_inv))(params={}))


class TestEquivalence:
    def test_matches_vectorwoodbury(self, model):
        """logL_refdelta == logL_vectorwoodbury exactly (f64), any Phi / Phi_ref."""
        rng = np.random.default_rng(1)
        for _ in range(5):
            phi = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, K)))
            np.testing.assert_allclose(_rd(model, phi, model["phi_ref"]),
                                       _vw(model, phi), rtol=0, atol=1e-6)

    def test_reference_independent(self, model):
        phi = model["phi_base"]
        vals = [_rd(model, phi, jnp.asarray(10.0 ** np.random.default_rng(s).uniform(-7, -5, (NP, K))))
                for s in range(4)]
        np.testing.assert_allclose(vals, vals[0], rtol=0, atol=1e-6)


@contextlib.contextmanager
def _working(dtype):
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax", working=jnp.float64)


def _graphs_varPhi(model):
    """Phi(da) = phi_base * 10**da off the reference, so the algebra does not fold
    to a compile-time constant -- exercises the real float32 path."""
    base, ref = model["phi_base"], model["phi_ref"]

    def getPhi(params):
        return base * 10.0 ** params["da"]
    getPhi.params = ["da"]

    Plive, Pref = mh.NoiseMatrix(getPhi), mh.NoiseMatrix(ref)
    gw = mh.vectorwoodbury(model["ys"], _solves(model), model["Fs"], Plive.make_inv)
    gr = mh.vectorwoodbury_refdelta(model["ys"], _solves(model), model["Fs"],
                                    Plive.make_inv, Pref.make_inv)
    return gw, gr


class TestFloat32:
    def test_f32_finite_and_no_worse(self, model):
        """In float32 the refdelta logL is finite, accurate to its f64 truth, and no
        worse than the direct path. NOTE: on this benign synthetic problem the
        *direct* path already inherits the Half-A f64 combine and a well-conditioned
        mu, so its error is ~1e-6 and there is little left for Half-B to remove. The
        decisive ~380x gain is a large-array / realistic-Phi phenomenon, measured in
        the refdelta precision table (and re-checked
        in the rung-3 real-model test), not on synthetic matrices."""
        gw, gr = _graphs_varPhi(model)
        p = {"da": 0.3}
        tw, tr = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        with _working(jnp.float32):
            w32, r32 = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        err_wood, err_rd = abs(w32 - tw), abs(r32 - tr)
        assert np.isfinite(r32)
        assert err_rd < 1e-3, err_rd                 # increment is O(1) -> f32-accurate
        assert err_rd < err_wood + 1e-5, (err_rd, err_wood)

    def test_f32_dtype_contract(self, model):
        """White-box: the batched GP-block Cholesky stays float32, while the
        f64-combine nodes (logL_ref, dQ, dLdet, final sum) are float64."""
        from discovery.metamatrix import Node
        _, gr = _graphs_varPhi(model)
        pruned = mm.prune_graph(mm.fold_constants(gr))
        dm = mm._dtype_map(pruned, jnp.float32)
        cho_dtypes = [dm[n] for n, node in pruned.items()
                      if isinstance(node, Node) and "cho_factor" in (node.description or "")]
        assert any(d == jnp.float32 for d in cho_dtypes), \
            "expected the live GP-block Cholesky to stay float32"
        for n, node in pruned.items():
            if isinstance(node, Node) and getattr(node, "f64_combine", False):
                assert dm[n] == jnp.float64, f"f64-combine node {n} not float64"
