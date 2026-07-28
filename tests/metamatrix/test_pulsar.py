"""Tier-1 parity table: single-pulsar PulsarLikelihood.

Each row builds the same model under three routes (matrix.py reference,
metamath-via-monkeypatch, metamath-native via likelihood_metamath.py) and
asserts the output method (logL / conditional / clogL / ...) and param set
agree across all three. See ``_routes.build_routes`` for the route
definitions. Model recipes are imported from ``discovery.recipes`` (shared with the
docs cookbook).
"""

import numpy as np
import pytest

import jax

import discovery as ds

from ._comparison import assert_close, assert_params_equal
from ._routes import build_routes
import discovery.recipes as R


# Model builders live in `discovery.recipes` (in discovery.recipes, shared with the docs cookbook); here we
# just select which recipes feed which parity assertions.

# ---------- logL rows ----------

LOGL_ROWS = [
    pytest.param(R.measurement_simple,    id="measurement_simple"),
    pytest.param(R.measurement_white,     id="measurement_white"),
    pytest.param(R.ecorr_gp,              id="measurement+ecorr_gp"),
    pytest.param(R.ecorr_sm,              id="ecorr_sm+timing"),
    pytest.param(R.meas_timing,           id="measurement+timing_svd"),
    pytest.param(R.full_rn,               id="full_rn"),
    pytest.param(R.full_rn_concat_false,  id="full_rn_concat_false"),
    pytest.param(R.multi_vgp,             id="multi_vgp"),
    pytest.param(R.variable_timing,       id="variable_timing"),
    pytest.param(R.fftcov_2d,             id="fftcov_2d"),
    pytest.param(R.delay,                 id="delay"),
    pytest.param(R.fourier_variance_fixed, id="fourier_variance_fixed"),
]


# ---------- helpers ----------

ALT_ROUTES = ("mh_patched", "mh_native")


def _routes(build, psr):
    """Build the model under each route. Returns the dict from
    ``build_routes``."""
    return build_routes(lambda: build(psr))


def _draw_params(model, seed=0):
    np.random.seed(seed)
    return ds.sample_uniform(model.logL.params)


# ---------- tests ----------

@pytest.mark.parametrize("build", LOGL_ROWS)
def test_logL(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]
    p0 = _draw_params(ref)
    ref_val = float(ref.logL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].logL, ref.logL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].logL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


# conditional / clogL only make sense with at least one variable GP

CONDITIONAL_ROWS = [
    pytest.param(R.full_rn, id="full_rn"),
    pytest.param(R.multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_conditional(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]
    p0 = _draw_params(ref)
    mu_ref, cf_ref = ref.conditional(p0)

    for route in ALT_ROUTES:
        assert_params_equal(r[route].conditional, ref.conditional,
                            name=f"{build.__name__}[{route}]")
        mu, cf = r[route].conditional(p0)
        assert_close(np.asarray(mu), np.asarray(mu_ref), kind="coeffs",
                     name=f"{build.__name__}[{route}].mu")
        assert_close(np.asarray(cf[0]), np.asarray(cf_ref[0]), kind="matrix",
                     name=f"{build.__name__}[{route}].cf")


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_clogL(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    # clogL params include latent coefficient vectors like "..._coefficients(60)"
    # for which there is no standard prior; draw those manually.
    np.random.seed(0)
    scalar_params = [p for p in ref.clogL.params if not p.endswith(")")]
    p0 = ds.sample_uniform(scalar_params)
    for p in ref.clogL.params:
        if p.endswith(")"):
            n = int(p[p.index("(") + 1: -1])
            p0[p] = 1e-6 * np.random.randn(n)
    ref_val = float(ref.clogL(p0))

    for route in ALT_ROUTES:
        assert_params_equal(r[route].clogL, ref.clogL,
                            name=f"{build.__name__}[{route}]")
        val = float(r[route].clogL(p0))
        assert_close(val, ref_val, kind="logL",
                     name=f"{build.__name__}[{route}]")


SAMPLE_ROWS = [
    pytest.param(R.full_rn, id="full_rn"),
    pytest.param(R.multi_vgp, id="multi_vgp"),
]


@pytest.mark.parametrize("build", SAMPLE_ROWS)
def test_sample(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    p0 = _draw_params(ref)
    key = jax.random.PRNGKey(42)
    _, y_ref = ref.sample(key, p0)

    for route in ALT_ROUTES:
        _, y = r[route].sample(key, p0)
        assert_close(np.asarray(y), np.asarray(y_ref), kind="residuals",
                     name=f"{build.__name__}[{route}]")


@pytest.mark.parametrize("build", CONDITIONAL_ROWS)
def test_sample_conditional(psr, build):
    r = _routes(build, psr)
    ref = r["matrix"]

    p0 = _draw_params(ref)
    key = jax.random.PRNGKey(42)
    _, c_ref = ref.sample_conditional(key, p0)

    for route in ALT_ROUTES:
        _, c = r[route].sample_conditional(key, p0)
        assert set(c.keys()) == set(c_ref.keys())
        for k in c_ref:
            assert_close(np.asarray(c[k]), np.asarray(c_ref[k]),
                         kind="coeffs",
                         name=f"{build.__name__}[{route}].{k}")
