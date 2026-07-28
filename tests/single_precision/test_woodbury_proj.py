"""Test rung for the timing-model projection graph ``mh.woodbury_proj``.

The projection computes the marginal logL for the model

    y = M eps + F a + n,   n ~ N(0, K),   a ~ N(0, Phi),   eps flat (improper)

by whitening with K = diag(N) + ECORR and orthogonally projecting the data and
the Fourier basis F out of the span of the (whitened) timing design M -- the
float32-safe alternative to giving eps a huge-variance (1e40) Gaussian prior and
running the ordinary Woodbury. See
docs/metamatrix_dev.md (timing-model projection).

Equivalence (the f64 oracle): the flat-prior limit is exactly the proper
1e40-prior Woodbury minus the diverging prior-normalisation constant. Under
discovery's logL convention (no n*log(2pi) term, and the dimension count is the
same on both sides), the only difference is the prior logdet block:

    logL_proj  ==  logL_woodbury  +  0.5 * m_tm * log(sigma_eps^2)

with sigma_eps^2 = 1e40. We check this for both noise models (ECORR via
``smwhiten`` and pure-diagonal via ``dwhiten``), check that the offset is
genuinely a data-independent constant, and smoke-test that the projection stays
finite under a float32 working dtype (it never forms the 1e40 block).
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


SIGMA2 = 1.0e40          # the production timing-model prior variance (signals.py)


def _exposure(n_toa, n_epoch, seed):
    """Random 0/1 ECORR exposure matrix, every epoch hit at least once."""
    rng = np.random.default_rng(seed)
    epoch_of = rng.integers(0, n_epoch, size=n_toa)
    F = np.zeros((n_toa, n_epoch))
    F[np.arange(n_toa), epoch_of] = 1.0
    for j in range(n_epoch):
        if F[:, j].sum() == 0:
            F[rng.integers(0, n_toa), j] = 1.0
    return np.asarray(F, dtype=int)


@pytest.fixture(scope="module")
def model():
    """A single-pulsar timing + GP model: M (timing), F (Fourier GP), Phi, plus
    both an ECORR and a diagonal noise kernel over the same residuals."""
    n_toa, n_epoch, m_tm, k = 60, 10, 3, 4
    rng = np.random.default_rng(0)

    N_arr = jnp.asarray(rng.uniform(0.5, 2.0, n_toa))
    F_ec = _exposure(n_toa, n_epoch, 1)
    P_ec = jnp.asarray(rng.uniform(0.1, 1.0, n_epoch))
    M = jnp.asarray(rng.standard_normal((n_toa, m_tm)))      # timing design
    Fgp = jnp.asarray(rng.standard_normal((n_toa, k)))       # Fourier GP basis
    phi = jnp.asarray(rng.uniform(0.5, 3.0, k))              # GP prior variance
    y = jnp.asarray(rng.standard_normal(n_toa))
    y2 = jnp.asarray(rng.standard_normal(n_toa))             # second dataset

    N_sm = mh.NoiseMatrixSM(N_arr, F_ec, P_ec)               # ECORR noise
    N_diag = mh.NoiseMatrix(N_arr)                           # diagonal noise

    # Combined basis + full (timing+GP) prior for the ordinary-Woodbury oracle.
    Fc = jnp.concatenate([M, Fgp], axis=1)
    phi_full = jnp.concatenate([jnp.full(m_tm, SIGMA2), phi])
    P_full = mh.NoiseMatrix(phi_full)   # make_inv -> (diag(1/phi), sum log phi)
    P_gp = mh.NoiseMatrix(phi)          # GP-only prior for the projection

    return dict(m_tm=m_tm, y=y, y2=y2, M=M, Fgp=Fgp, Fc=Fc,
                N_sm=N_sm, N_diag=N_diag, P_full=P_full, P_gp=P_gp)


def _woodbury_logL(y, N, Fc, P_full):
    return float(mm.func(mh.woodbury(y, N.make_solve, Fc, P_full.make_inv))(params={}))


def _proj_logL(y, N, M, Fgp, P_gp):
    return float(mm.func(mh.woodbury_proj(y, N.make_whiten, M, Fgp, P_gp.make_inv))(params={}))


class TestProjectionEquivalence:
    """logL_proj == logL_woodbury(1e40 prior) + 0.5 m_tm log(1e40), to f64 tol."""

    def test_ecorr_noise(self, model):
        m = model
        const = 0.5 * m["m_tm"] * np.log(SIGMA2)
        wb = _woodbury_logL(m["y"], m["N_sm"], m["Fc"], m["P_full"])
        pr = _proj_logL(m["y"], m["N_sm"], m["M"], m["Fgp"], m["P_gp"])
        np.testing.assert_allclose(pr, wb + const, rtol=1e-8, atol=1e-6)

    def test_diagonal_noise(self, model):
        """Exercises the diagonal whitening emitter ``dwhiten`` (no ECORR)."""
        m = model
        const = 0.5 * m["m_tm"] * np.log(SIGMA2)
        wb = _woodbury_logL(m["y"], m["N_diag"], m["Fc"], m["P_full"])
        pr = _proj_logL(m["y"], m["N_diag"], m["M"], m["Fgp"], m["P_gp"])
        np.testing.assert_allclose(pr, wb + const, rtol=1e-8, atol=1e-6)

    def test_offset_is_data_independent(self, model):
        """The proj - woodbury gap must be a pure prior constant: identical for a
        second, unrelated dataset (confirms it is not a data-dependent error)."""
        m = model
        gap1 = (_proj_logL(m["y"], m["N_sm"], m["M"], m["Fgp"], m["P_gp"])
                - _woodbury_logL(m["y"], m["N_sm"], m["Fc"], m["P_full"]))
        gap2 = (_proj_logL(m["y2"], m["N_sm"], m["M"], m["Fgp"], m["P_gp"])
                - _woodbury_logL(m["y2"], m["N_sm"], m["Fc"], m["P_full"]))
        np.testing.assert_allclose(gap1, gap2, rtol=1e-8, atol=1e-6)


@contextlib.contextmanager
def _working(dtype):
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax", working=jnp.float64)


def _proj_graph_varN(model):
    """Build woodbury_proj with a parameter-dependent N (efac) so the linear
    algebra does NOT fold to a compile-time constant -- needed to exercise the
    real float32 eval path and inspect the working-dtype map."""
    m = model
    src = m["N_sm"]

    def getN(params):
        return params["efac"] ** 2 * src.N
    getN.params = ["efac"]

    N_sm = mh.NoiseMatrixSM(getN, src.F, src.P)   # rebuild with param-dependent N
    return mh.woodbury_proj(m["y"], N_sm.make_whiten, m["M"], m["Fgp"],
                            m["P_gp"].make_inv)


class TestProjectionFloat32:
    """The projection never forms the 1e40 prior block, so under a float32
    working dtype it stays finite and close to its own f64 value -- unlike the
    1e40 Woodbury, whose timing prior block underflows/loses conditioning in f32.
    (The full NG15-scale 1.62 -> small win is exercised by the array harness, not
    here; this is the single-pulsar finiteness/closeness smoke test.)"""

    def test_proj_finite_and_close_in_f32(self, model):
        g = _proj_graph_varN(model)
        f = mm.func(g)
        ref = float(f(params={"efac": 1.0}))         # default f64
        with _working(jnp.float32):
            f32 = float(mm.func(g)(params={"efac": 1.0}))
        assert np.isfinite(f32)
        # single-pulsar scale: the f64 pins keep it tight; loose bar just guards
        # against a gross dtype regression.
        np.testing.assert_allclose(f32, ref, rtol=1e-3, atol=1e-2)

    def test_f32_dtype_contract(self, model):
        """White-box: under working=float32 the *expensive* GP-block Cholesky
        runs in float32, while the pinned terms (ytNmy, lN, lP) and the final
        logL combination run in float64. The small timing-Gram factorization is
        pulled to float64 too (it is an ancestor of the pinned ytNmy via the
        projected residual) -- cheap and desirable, since that subtraction is the
        cancellation-prone step."""
        from discovery.metamatrix import Node
        pruned = mm.prune_graph(mm.fold_constants(_proj_graph_varN(model)))
        dm = mm._dtype_map(pruned, jnp.float32)

        cho_dtypes = [dm[n] for n, node in pruned.items()
                      if isinstance(node, Node) and "cho_factor" in (node.description or "")]
        assert any(d == jnp.float32 for d in cho_dtypes), \
            "expected the GP-block Cholesky to stay float32"

        # every pin / f64-combine node must be float64
        for n, node in pruned.items():
            if isinstance(node, Node) and (getattr(node, "pin", False)
                                           or getattr(node, "f64_combine", False)):
                assert dm[n] == jnp.float64, f"pinned node {n} not float64"
