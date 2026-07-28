"""Rung-1 test for the fused two-level reference+delta: the INNER half,
``mh.vectorwoodburyjointsolve_refdelta``.

The fused inner solve (``vectorwoodburyjointsolve``) projects each per-pulsar
intrinsic-red-noise (IRN) GP out and hands the outer (global HD/CURN) level the
projected quantities (a~, b~, G~) plus the per-pulsar inner logdet. The refdelta
twin instead emits, off a frozen inner reference Phi_ref,in:

  * the REFERENCE projected quantities (a~_ref, b~_ref, G~_ref), and
  * the per-call INCREMENTS (Delta a~, Delta b~, Delta G~) formed directly via the
    resolvent identity (note sec. 2-3) -- never as current-minus-reference.

This test locks the inner algebra: reconstructing current = reference + increment
must reproduce, in float64, the trusted direct ``vectorwoodburyjointsolve``
projected output (any Phi, any Phi_ref -- the decomposition is exact). The inner
merged logdet sum_i logdet(I + Phi_in,i G_in,i) is cross-checked against numpy.

See research_note_nested_increment.md (sec. 2-4) and
fused reference+delta (rung 1); see docs/metamatrix_dev.md.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from discovery import metamath as mh
from discovery import metamatrix as mm

NP, M_IN, M_OUT = 3, 5, 4
NTOA = [120, 150, 90]


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    ys = [jnp.asarray(20.0 * rng.standard_normal(n)) for n in NTOA]
    Ns = [mh.NoiseMatrix(jnp.asarray(rng.uniform(0.5, 2.0, n))) for n in NTOA]
    Fs_in = [jnp.asarray(rng.standard_normal((n, M_IN))) for n in NTOA]
    Fs_out = [jnp.asarray(rng.standard_normal((n, M_OUT))) for n in NTOA]
    phi_in = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, M_IN)))
    phi_in_ref = jnp.asarray(10.0 ** rng.uniform(-7.0, -5.0, (NP, M_IN)))
    return dict(ys=ys, Ns=Ns, Fs_in=Fs_in, Fs_out=Fs_out,
                phi_in=phi_in, phi_in_ref=phi_in_ref)


def _direct_projected(m, phi_in):
    """Trusted current (a~, b~, G~) from the existing fused inner solve."""
    P = mh.NoiseMatrix(phi_in)
    g = mh.vectorwoodburyjointsolve(m["ys"], m["Fs_out"], [N.make_solve for N in m["Ns"]],
                                    m["Fs_in"], P.make_inv)
    ytNmy_proj, ld, FtNmy_proj, FtNmF_proj = mm.func(mm.prune_graph(g, output='projected'))(params={})
    return dict(atil=np.asarray(ytNmy_proj), btil=np.asarray(FtNmy_proj),
                Gtil=np.asarray(FtNmF_proj), ld=np.asarray(ld))


def _refdelta(m, phi_in, phi_in_ref):
    P, Pr = mh.NoiseMatrix(phi_in), mh.NoiseMatrix(phi_in_ref)
    g = mh.vectorwoodburyjointsolve_refdelta(
        m["ys"], m["Fs_out"], [N.make_solve for N in m["Ns"]], m["Fs_in"],
        P.make_inv, Pr.make_inv)
    # rung 1 now emits two separate outputs: 'refconst' (constant reference
    # quantities, fold) and 'refincr' (live increments). Prune each and merge.
    const = mm.func(mm.prune_graph(g, output='refconst'))(params={})
    incr = mm.func(mm.prune_graph(g, output='refincr'))(params={})
    out = {k: np.asarray(v) for k, v in
           zip(['aref_sum', 'ld_in_ref', 'btil_ref', 'Gtil_ref', 'lN_sum'], const)}
    out.update({k: np.asarray(v) for k, v in
                zip(['dA', 'd_ld_in', 'dbtil', 'dGtil'], incr)})
    return out


def _inner_logdet_numpy(m, phi_in):
    """sum_i logdet(I + Phi_in,i G_in,i) directly from the inputs."""
    total = 0.0
    phi_in = np.asarray(phi_in)
    for i in range(NP):
        Ninv = 1.0 / np.asarray(m["Ns"][i].N)
        Fin = np.asarray(m["Fs_in"][i])
        G_in = Fin.T @ (Ninv[:, None] * Fin)
        total += np.linalg.slogdet(np.eye(M_IN) + np.diag(phi_in[i]) @ G_in)[1]
    return total


class TestInnerRefdelta:
    def test_reconstructs_direct_projected(self, problem):
        """current = reference + increment must equal the direct projected output
        (f64), for several Phi moves off the reference."""
        rng = np.random.default_rng(1)
        for _ in range(4):
            phi_in = problem["phi_in_ref"] * 10.0 ** jnp.asarray(rng.uniform(-1.0, 1.0, (NP, M_IN)))
            d = _direct_projected(problem, phi_in)
            r = _refdelta(problem, phi_in, problem["phi_in_ref"])

            np.testing.assert_allclose(r["aref_sum"] + r["dA"], d["atil"].sum(), rtol=0, atol=1e-7)
            np.testing.assert_allclose(r["btil_ref"] + r["dbtil"], d["btil"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(r["Gtil_ref"] + r["dGtil"], d["Gtil"], rtol=0, atol=1e-8)

    def test_inner_logdet_matches_numpy(self, problem):
        """ld_in_ref + d_ld_in reconstructs sum_i logdet(I + Phi_in G_in)."""
        rng = np.random.default_rng(2)
        phi_in = problem["phi_in_ref"] * 10.0 ** jnp.asarray(rng.uniform(-1.0, 1.0, (NP, M_IN)))
        r = _refdelta(problem, phi_in, problem["phi_in_ref"])
        np.testing.assert_allclose(r["ld_in_ref"], _inner_logdet_numpy(problem, problem["phi_in_ref"]),
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(r["ld_in_ref"] + r["d_ld_in"], _inner_logdet_numpy(problem, phi_in),
                                   rtol=0, atol=1e-7)

    def test_increment_vanishes_at_reference(self, problem):
        """At Phi = Phi_ref every increment is zero; reference == direct."""
        r = _refdelta(problem, problem["phi_in_ref"], problem["phi_in_ref"])
        for k in ["dA", "d_ld_in"]:
            np.testing.assert_allclose(r[k], 0.0, atol=1e-9)
        np.testing.assert_allclose(r["dbtil"], 0.0, atol=1e-9)
        np.testing.assert_allclose(r["dGtil"], 0.0, atol=1e-9)
