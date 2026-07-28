"""Rung-2 test for the fused two-level reference+delta: the OUTER half,
``mh.globalwoodbury_fused_refdelta``.

The fused path is a two-level nested Woodbury: a per-pulsar intrinsic-red-noise
(IRN) inner GP, then a dense cross-pulsar (HD / CURN) outer GP. Rung 1
(``vectorwoodburyjointsolve_refdelta``) emits the inner reference projected
quantities + increments; rung 2 (this) does the outer two-perturbation update and
assembles logL = logL_ref + Delta logL (note sec. 4-5).

The test drives BOTH rungs and locks the assembly two ways:

  * **exactness (f64):** logL_refdelta must equal the trusted direct fused path
    (``vectorwoodburyjointsolve`` -> ``globalwoodbury_fused``) to f64, for inner-only,
    outer-only, and combined Phi moves off the reference -- the decomposition is exact;
  * **reference-independence:** the value does not depend on the frozen references;
  * **float32:** the increment is O(1) so the refdelta logL stays accurate in float32
    while the expensive *current* outer Cholesky remains float32 (the speed win).

The increment *formula itself* is independently locked against an mpmath/brute-force
oracle in test_refdelta_nested.py. See research_note_nested_increment.md (sec. 4-5)
and piece2_fused_refdelta_plan.md (rung 2).
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

NP, M_IN, M_OUT = 4, 5, 3
NTOA = [300 + 31 * i for i in range(NP)]
NGW = NP * M_OUT


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    ys = [jnp.asarray(20.0 * rng.standard_normal(n)) for n in NTOA]
    Ns = [mh.NoiseMatrix(jnp.asarray(rng.uniform(0.5, 2.0, n))) for n in NTOA]
    Fs_in = [jnp.asarray(rng.standard_normal((n, M_IN))) for n in NTOA]
    Fs_out = [jnp.asarray(rng.standard_normal((n, M_OUT))) for n in NTOA]
    phi_in_ref = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, M_IN)))
    phi_gw_ref = jnp.asarray(_spd(rng, NGW))
    return dict(ys=ys, Ns=Ns, Fs_in=Fs_in, Fs_out=Fs_out,
                phi_in_ref=phi_in_ref, phi_gw_ref=phi_gw_ref)


def _spd(rng, n, scale=1e-6):
    """A well-conditioned SPD dense outer covariance (HD-like)."""
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T / n + 3.0 * np.eye(n))


def _solves(m):
    return [N.make_solve for N in m["Ns"]]


def _fused(m, phi_in, phi_gw):
    """Trusted direct fused logL (inner jointsolve -> outer globalwoodbury_fused)."""
    Pin, Pgw = mh.NoiseMatrix(phi_in), mh.NoiseMatrix(phi_gw)
    joint = mh.vectorwoodburyjointsolve(m["ys"], m["Fs_out"], _solves(m), m["Fs_in"], Pin.make_inv)
    proj = mm.prune_graph(joint, output='projected')
    return float(mm.func(mh.globalwoodbury_fused(proj, Pgw.make_inv))(params={}))


def _fused_rd_graph(m, phi_in, phi_gw, phi_in_ref, phi_gw_ref):
    Pin, Pin_r = mh.NoiseMatrix(phi_in), mh.NoiseMatrix(phi_in_ref)
    Pgw, Pgw_r = mh.NoiseMatrix(phi_gw), mh.NoiseMatrix(phi_gw_ref)
    joint = mh.vectorwoodburyjointsolve_refdelta(
        m["ys"], m["Fs_out"], _solves(m), m["Fs_in"], Pin.make_inv, Pin_r.make_inv)
    # rung 1 emits two outputs now: constant reference group + live increment group
    refconst = mm.prune_graph(joint, output='refconst')
    refincr = mm.prune_graph(joint, output='refincr')
    return mh.globalwoodbury_fused_refdelta(refconst, refincr, Pgw.make_inv, Pgw_r.make_inv)


def _fused_rd(m, phi_in, phi_gw, phi_in_ref, phi_gw_ref):
    return float(mm.func(_fused_rd_graph(m, phi_in, phi_gw, phi_in_ref, phi_gw_ref))(params={}))


def _moves(m, seed):
    """Inner (diagonal) and outer (dense SPD) Phi off the references."""
    rng = np.random.default_rng(seed)
    phi_in = m["phi_in_ref"] * 10.0 ** jnp.asarray(rng.uniform(-1.0, 1.0, (NP, M_IN)))
    phi_gw = jnp.asarray(_spd(rng, NGW))
    return phi_in, phi_gw


class TestExactness:
    def test_matches_direct_fused(self, model):
        """logL_refdelta == direct fused logL (f64) for inner/outer/both moves."""
        # both move
        for seed in range(4):
            phi_in, phi_gw = _moves(model, seed)
            np.testing.assert_allclose(
                _fused_rd(model, phi_in, phi_gw, model["phi_in_ref"], model["phi_gw_ref"]),
                _fused(model, phi_in, phi_gw), rtol=0, atol=1e-5)
        # inner only (outer at reference)
        phi_in, _ = _moves(model, 10)
        np.testing.assert_allclose(
            _fused_rd(model, phi_in, model["phi_gw_ref"], model["phi_in_ref"], model["phi_gw_ref"]),
            _fused(model, phi_in, model["phi_gw_ref"]), rtol=0, atol=1e-5)
        # outer only (inner at reference)
        _, phi_gw = _moves(model, 11)
        np.testing.assert_allclose(
            _fused_rd(model, model["phi_in_ref"], phi_gw, model["phi_in_ref"], model["phi_gw_ref"]),
            _fused(model, model["phi_in_ref"], phi_gw), rtol=0, atol=1e-5)

    def test_vanishes_at_reference(self, model):
        """Phi = Phi_ref -> Delta logL = 0, value == direct fused at the reference."""
        np.testing.assert_allclose(
            _fused_rd(model, model["phi_in_ref"], model["phi_gw_ref"],
                      model["phi_in_ref"], model["phi_gw_ref"]),
            _fused(model, model["phi_in_ref"], model["phi_gw_ref"]), rtol=0, atol=1e-6)

    def test_reference_independent(self, model):
        """The logL does not depend on the frozen references."""
        phi_in, phi_gw = _moves(model, 7)
        vals = []
        for seed in range(3):
            rng = np.random.default_rng(100 + seed)
            phi_in_ref = model["phi_in_ref"] * 10.0 ** jnp.asarray(rng.uniform(-0.5, 0.5, (NP, M_IN)))
            phi_gw_ref = jnp.asarray(_spd(rng, NGW))
            vals.append(_fused_rd(model, phi_in, phi_gw, phi_in_ref, phi_gw_ref))
        np.testing.assert_allclose(vals, vals[0], rtol=0, atol=1e-5)


@contextlib.contextmanager
def _working(dtype):
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax", working=jnp.float64)


def _graphs_varPhi(model):
    """Phi(da) = phi_in_ref * 10**da (inner) and phi_gw_ref * (1+dg) (outer) so the
    algebra does not fold to a compile-time constant -- exercises the real f32 path."""
    base_in, ref_in = model["phi_in_ref"], model["phi_in_ref"]
    ref_gw = model["phi_gw_ref"]

    def getPhi_in(params):
        return base_in * 10.0 ** params["da"]
    getPhi_in.params = ["da"]

    def getPhi_gw(params):
        return ref_gw * (1.0 + params["dg"])
    getPhi_gw.params = ["dg"]

    Pin, Pin_r = mh.NoiseMatrix(getPhi_in), mh.NoiseMatrix(ref_in)
    Pgw, Pgw_r = mh.NoiseMatrix(getPhi_gw), mh.NoiseMatrix(ref_gw)

    joint_w = mh.vectorwoodburyjointsolve(model["ys"], model["Fs_out"], _solves(model),
                                          model["Fs_in"], Pin.make_inv)
    gw = mh.globalwoodbury_fused(mm.prune_graph(joint_w, output='projected'), Pgw.make_inv)

    joint_r = mh.vectorwoodburyjointsolve_refdelta(model["ys"], model["Fs_out"], _solves(model),
                                                   model["Fs_in"], Pin.make_inv, Pin_r.make_inv)
    gr = mh.globalwoodbury_fused_refdelta(mm.prune_graph(joint_r, output='refconst'),
                                          mm.prune_graph(joint_r, output='refincr'),
                                          Pgw.make_inv, Pgw_r.make_inv)
    return gw, gr


class TestFloat32:
    def test_f32_finite_and_no_worse(self, model):
        """In float32 the fused refdelta logL is finite, accurate to its f64 truth,
        and no worse than the direct fused path."""
        gw, gr = _graphs_varPhi(model)
        p = {"da": 0.3, "dg": 0.2}
        tw, tr = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        with _working(jnp.float32):
            w32, r32 = float(mm.func(gw)(params=p)), float(mm.func(gr)(params=p))
        err_wood, err_rd = abs(w32 - tw), abs(r32 - tr)
        assert np.isfinite(r32)
        assert err_rd < 1e-2, err_rd
        assert err_rd < err_wood + 1e-5, (err_rd, err_wood)

    def test_f32_dtype_contract(self, model):
        """White-box: the current outer GP-block Cholesky stays float32, while the
        f64-combine nodes (logL_ref, dQ, final sum) are float64."""
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
