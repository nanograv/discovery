"""Parity tests for the graph-decomposed `mh.smsolve` against the original
`matrix.SM_1d_indexed` / `matrix.SM_2d_indexed` oracles.

The graph version exposes `sum(log N)`, `pad(N)`, and the
`sum log1p(P · F^T diag(1/N) F)` precompute as separate nodes so
`fold_constants` can bake them when N (and/or P) are constant. Output
must agree with the oracle to machine precision in all const/var
combinations.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from discovery import matrix
from discovery import metamath as mh
from discovery import metamatrix as mm


def _make_exposure(n_toa=40, n_epoch=8, seed=0):
    """Build a random 0/1 exposure matrix F (n_toa, n_epoch) plus Uind."""
    rng = np.random.default_rng(seed)
    epoch_of = rng.integers(0, n_epoch, size=n_toa)
    F = np.zeros((n_toa, n_epoch))
    F[np.arange(n_toa), epoch_of] = 1.0
    # ensure every epoch has at least one TOA
    for j in range(n_epoch):
        if F[:, j].sum() == 0:
            F[rng.integers(0, n_toa), j] = 1.0
    Uind = matrix.make_uind(np.asarray(F, dtype=int))
    return jnp.asarray(F), jnp.asarray(Uind)


def _ref_1d(y, N, Uind, P):
    return matrix.SM_1d_indexed(y, N, Uind, P)


def _ref_2d(Y2, N, Uind, P):
    # mh._sm_apply takes Y in (d, k) and returns (d, k). matrix.SM_2d_indexed
    # expects (k, d) and returns (k, d) + scalar. Match the mh contract.
    KmT, lK = matrix.SM_2d_indexed(Y2.T, N, Uind, P)
    return KmT.T, lK


@pytest.fixture(scope="module")
def sm_setup():
    F, Uind = _make_exposure(n_toa=40, n_epoch=8, seed=0)
    rng = np.random.default_rng(1)
    N = jnp.asarray(rng.uniform(0.5, 2.0, size=F.shape[0]))
    P = jnp.asarray(rng.uniform(0.1, 1.0, size=F.shape[1]))
    y = jnp.asarray(rng.standard_normal(F.shape[0]))
    Y2 = jnp.asarray(rng.standard_normal((F.shape[0], 5)))
    return dict(F=F, Uind=Uind, N=N, P=P, y=y, Y2=Y2)


def _eval_smsolve(y, N, Uind, P):
    """Build smsolve graph for the given operand shapes and evaluate it."""
    graph = mh.smsolve(y, N, Uind, P)
    f = mm.func(graph)
    return f(params={})


class TestSMSolveParity:
    def test_1d_all_const(self, sm_setup):
        s = sm_setup
        Kmy_ref, ld_ref = _ref_1d(s["y"], s["N"], s["Uind"], s["P"])
        Kmy, ld = _eval_smsolve(s["y"], s["N"], s["Uind"], s["P"])
        np.testing.assert_allclose(Kmy, Kmy_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ld, ld_ref, rtol=1e-12, atol=1e-12)

    def test_2d_all_const(self, sm_setup):
        s = sm_setup
        Kmy_ref, ld_ref = _ref_2d(s["Y2"], s["N"], s["Uind"], s["P"])
        Kmy, ld = _eval_smsolve(s["Y2"], s["N"], s["Uind"], s["P"])
        np.testing.assert_allclose(Kmy, Kmy_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ld, ld_ref, rtol=1e-12, atol=1e-12)

    def test_1d_var_N(self, sm_setup):
        """N comes in as a FuncLeaf (param-dependent); should still match oracle."""
        s = sm_setup
        N_arr = s["N"]

        def getN(params):
            return params["efac"] ** 2 * N_arr
        getN.params = ["efac"]

        graph = mh.smsolve(s["y"], getN, s["Uind"], s["P"])
        f = mm.func(graph)
        for efac in (1.0, 1.3, 0.7):
            Kmy, ld = f(params={"efac": efac})
            Kmy_ref, ld_ref = _ref_1d(s["y"], efac ** 2 * N_arr,
                                      s["Uind"], s["P"])
            np.testing.assert_allclose(Kmy, Kmy_ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ld, ld_ref, rtol=1e-12, atol=1e-12)

    def test_1d_var_P(self, sm_setup):
        """P param-dependent (varying ecorr amplitude)."""
        s = sm_setup
        P_arr = s["P"]

        def getP(params):
            return params["log10_ecorr"] * P_arr
        getP.params = ["log10_ecorr"]

        graph = mh.smsolve(s["y"], s["N"], s["Uind"], getP)
        f = mm.func(graph)
        for amp in (0.5, 1.0, 2.5):
            Kmy, ld = f(params={"log10_ecorr": amp})
            Kmy_ref, ld_ref = _ref_1d(s["y"], s["N"], s["Uind"], amp * P_arr)
            np.testing.assert_allclose(Kmy, Kmy_ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ld, ld_ref, rtol=1e-12, atol=1e-12)

    def test_1d_var_N_and_P(self, sm_setup):
        s = sm_setup
        N_arr, P_arr = s["N"], s["P"]

        def getN(params):
            return params["efac"] ** 2 * N_arr
        getN.params = ["efac"]

        def getP(params):
            return params["log10_ecorr"] * P_arr
        getP.params = ["log10_ecorr"]

        graph = mh.smsolve(s["y"], getN, s["Uind"], getP)
        f = mm.func(graph)
        Kmy, ld = f(params={"efac": 1.2, "log10_ecorr": 0.7})
        Kmy_ref, ld_ref = _ref_1d(s["y"], 1.2 ** 2 * N_arr,
                                  s["Uind"], 0.7 * P_arr)
        np.testing.assert_allclose(Kmy, Kmy_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ld, ld_ref, rtol=1e-12, atol=1e-12)


class TestSMSolveFolding:
    """When N and P are both constant, all of logN, Np, log1pt, logdet
    should collapse to ConstLeafs at trace time; only the y-dependent
    `Kmy` node remains. This is the architectural payoff."""

    def test_all_const_folds_logdet(self, sm_setup):
        s = sm_setup
        graph = mh.smsolve(None, s["N"], s["Uind"], s["P"])  # y is ArgLeaf
        folded = mm.fold_constants(graph)

        # Identify the y arg leaf and the result pair.
        from discovery.metamatrix import ArgLeaf, ConstLeaf, Node
        n_arg = sum(isinstance(v, ArgLeaf) for v in folded.values())
        n_node = sum(isinstance(v, Node) for v in folded.values())
        assert n_arg == 1, f"expected 1 ArgLeaf (y), got {n_arg}"

        # The logdet sub-expression should fold completely. After pruning to
        # `result`, only the y-dependent path + the const logdet + the pair
        # node remain. We expect a small number of runtime Nodes (the apply
        # for Kmy plus the pair packaging), and the rest ConstLeaf.
        assert n_node <= 3, (
            f"expected ≤3 runtime nodes when N,P const; got {n_node}. "
            f"Folding likely failed to bake logN / Np / log1pt."
        )

    def test_var_N_keeps_more_runtime(self, sm_setup):
        """When N is variable, logN / Np / log1pt cannot fold."""
        s = sm_setup

        def getN(params):
            return params["efac"] ** 2 * s["N"]
        getN.params = ["efac"]

        graph = mh.smsolve(None, getN, s["Uind"], s["P"])
        folded = mm.fold_constants(graph)
        from discovery.metamatrix import Node
        n_node = sum(isinstance(v, Node) for v in folded.values())
        # Strictly more runtime nodes than the all-const case (logN, Np,
        # log1pt, the sum, and Kmy + pair survive).
        assert n_node >= 4
