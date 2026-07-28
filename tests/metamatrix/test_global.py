"""Tier-2 parity table: GlobalLikelihood.

GlobalLikelihood wraps a list of PulsarLikelihoods plus optional globalgp
(correlated GP across pulsars). With the monkeypatch, globalgp.Phi becomes
mh.NoiseMatrix, which triggers the metamath GlobalWoodburyKernel path in
likelihood.py:307.
"""

import numpy as np
import pytest

import jax

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._routes import build_routes
import discovery.recipes as R


# Model builders live in `discovery.recipes` (in discovery.recipes, shared with the docs cookbook).

# ---------- tables ----------

LOGL_ROWS = [
    pytest.param(R.no_global,       id="no_global"),
    pytest.param(R.global_hd,       id="global_hd"),
    pytest.param(R.global_monopole, id="global_monopole"),
    pytest.param(R.global_compound, id="global_compound"),
]


# ---------- helpers ----------

ALT_ROUTES = ("mh_patched", "mh_native")


def _routes(build, psrs):
    return build_routes(lambda: build(psrs))


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psrs, build):
    r = _routes(build, psrs)
    ref = r["matrix"]
    np.random.seed(0)
    p0 = ds.sample_uniform(ref.logL.params)
    ref_val = float(ref.logL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].logL, ref.logL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].logL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


# conditional only meaningful with globalgp
CONDITIONAL_ROWS = [
    pytest.param(R.global_hd,       id="global_hd"),
    pytest.param(R.global_monopole, id="global_monopole"),
    pytest.param(R.global_compound, id="global_compound"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional(psrs, build):
    """GlobalLikelihood.conditional has its own bespoke matrix.py path; with
    monkeypatch, globalgp.Phi.make_inv etc still resolve through metamath.
    """
    r = _routes(build, psrs)
    ref = r["matrix"]
    np.random.seed(0)
    p0 = ds.sample_uniform(ref.conditional.params)
    mu_ref, cf_ref = ref.conditional(p0)

    for route in ALT_ROUTES:
        assert_params_equal(r[route].conditional, ref.conditional,
                            name=f"{build.__name__}[{route}]")
        mu, cf = r[route].conditional(p0)
        assert_close(np.asarray(mu), np.asarray(mu_ref), kind="coeffs",
                     name=f"{build.__name__}[{route}].mu")
        assert_close(np.asarray(cf[0]), np.asarray(cf_ref[0]), kind="matrix",
                     name=f"{build.__name__}[{route}].cf")


SAMPLE_ROWS = [
    pytest.param(R.no_global, id="no_global"),
    pytest.param(R.global_hd, id="global_hd"),
]


@pytest.mark.parametrize("build", SAMPLE_ROWS)
def test_sample(psrs, build):
    """Per-pulsar prior draws plus (when globalgp set) a correlated Phi draw."""
    r = _routes(build, psrs)
    ref = r["matrix"]
    np.random.seed(0)
    p0 = ds.sample_uniform(ref.logL.params)
    key = jax.random.PRNGKey(42)
    _, ys_ref = ref.sample(key, p0)

    for route in ALT_ROUTES:
        _, ys = r[route].sample(key, p0)
        assert len(ys) == len(ys_ref) == len(psrs)
        for i, (yr, y) in enumerate(zip(ys_ref, ys)):
            assert_close(np.asarray(y), np.asarray(yr), kind="residuals",
                         name=f"{build.__name__}[{route}].psr{i}")
