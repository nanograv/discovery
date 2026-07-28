"""Test rung for the single-level reference+delta graph ``mh.woodbury_refdelta``
(Piece 2 'Half B'; see research_note_on_split_with_reference.md sec. 2-3 and
docs/adr/0001,0003).

The marginal logL is written as logL(theta) = logL_ref + Delta logL, with logL_ref
frozen in float64 at a reference covariance Phi_ref and only the small O(1)
increment computed per call. This:

  * equals the ordinary ``woodbury`` logL exactly in float64, for any Phi and any
    reference Phi_ref (the decomposition is algebraically exact);
  * is independent of the choice of Phi_ref (it is a pure reparametrisation);
  * in float32 holds the logL far more accurately than the direct ``woodbury``
    path, because the increment is formed without the ytNmy - FtNmy.mu
    cancellation of two ~1e6 numbers;
  * keeps the expensive GP-block Cholesky in float32 (the speed win) while the
    reference baseline and the final scalar combination are float64.
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


@pytest.fixture(scope="module")
def model():
    """Single-pulsar GP model: diagonal white noise N, Fourier basis F, a
    power-law-ish diagonal GP prior Phi, a reference Phi_ref, and a big residual
    y so |logL| ~ 1e5 (enough to expose the float32 cancellation)."""
    n, k = 600, 14
    rng = np.random.default_rng(0)
    N = jnp.asarray(rng.uniform(0.5, 2.0, n))
    F = jnp.asarray(rng.standard_normal((n, k)))
    y = jnp.asarray(25.0 * rng.standard_normal(n))
    phi_base = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, k))
    phi_ref = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, k))
    return dict(n=n, k=k, N=mh.NoiseMatrix(N), F=F, y=y,
                phi_base=phi_base, phi_ref=phi_ref)


def _wood(m, phi):
    P = mh.NoiseMatrix(phi)
    return float(mm.func(mh.woodbury(m["y"], m["N"].make_solve, m["F"], P.make_inv))(params={}))


def _rd(m, phi, phi_ref):
    P, Pr = mh.NoiseMatrix(phi), mh.NoiseMatrix(phi_ref)
    return float(mm.func(mh.woodbury_refdelta(
        m["y"], m["N"].make_solve, m["F"], P.make_inv, Pr.make_inv))(params={}))


class TestRefdeltaEquivalence:
    """logL_refdelta == logL_woodbury exactly (f64), for any Phi / any Phi_ref."""

    def test_matches_woodbury(self, model):
        rng = np.random.default_rng(1)
        for _ in range(6):
            phi = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, model["k"]))
            np.testing.assert_allclose(_rd(model, phi, model["phi_ref"]),
                                       _wood(model, phi), rtol=0, atol=1e-7)

    def test_reference_independent(self, model):
        """Same logL regardless of which Phi_ref we expand around."""
        phi = model["phi_base"]
        vals = [_rd(model, phi, jnp.asarray(10.0 ** np.random.default_rng(s).uniform(-7, -5, model["k"])))
                for s in range(4)]
        np.testing.assert_allclose(vals, vals[0], rtol=0, atol=1e-7)


@contextlib.contextmanager
def _working(dtype):
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax", working=jnp.float64)


def _graphs_varPhi(model):
    """Build woodbury and woodbury_refdelta with a parameter-dependent Phi (a
    log10 shift `da` off the reference) so the linear algebra does NOT fold to a
    compile-time constant -- needed to exercise the real float32 path."""
    base, ref = model["phi_base"], model["phi_ref"]

    def getPhi(params):
        return base * 10.0 ** params["da"]
    getPhi.params = ["da"]

    Plive, Pref = mh.NoiseMatrix(getPhi), mh.NoiseMatrix(ref)
    gw = mh.woodbury(model["y"], model["N"].make_solve, model["F"], Plive.make_inv)
    gr = mh.woodbury_refdelta(model["y"], model["N"].make_solve, model["F"],
                              Plive.make_inv, Pref.make_inv)
    return gw, gr


class TestRefdeltaFloat32:
    def test_f32_beats_direct(self, model):
        """At a typical move off the reference, the reference+delta logL is much
        closer to the f64 truth than the direct (Half-A) woodbury logL."""
        gw, gr = _graphs_varPhi(model)
        p = {"da": 0.3}
        tw, tr = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        with _working(jnp.float32):
            w32, r32 = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        err_wood, err_rd = abs(w32 - tw), abs(r32 - tr)
        assert np.isfinite(r32)
        # reference+delta should be at least 10x tighter than the direct path
        assert err_rd < 0.1 * err_wood + 1e-9, (err_rd, err_wood)

    def test_f32_dtype_contract(self, model):
        """White-box: the GP-block Cholesky stays float32 (speed), while the
        f64-combine nodes (logL_ref, dQ, the final logL_ref+dlnL) are float64."""
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
