"""Rung-3 test: the reference+delta opt-in wired end-to-end through the real
metamath ArrayLikelihood (single-precision Half B).

`ArrayLikelihood(..., reference=theta_ref)` freezes each GP level's prior
covariance Phi ONCE at theta_ref in float64 (the "thin top layer",
`_freeze_reference`) and routes the marginal logL to the refdelta twins:

  * commongp only (CURN / IRN, no Hellings-Downs) -> single-level
    ``vectorwoodbury_refdelta``;
  * commongp + globalgp HD -> the two-level fused twins
    (``vectorwoodburyjointsolve_refdelta`` -> ``globalwoodbury_fused_refdelta``).

This locks the boundary, not the kernel algebra (that is rungs 1-2 + the oracle):

  * **opt-in is exact (f64):** logL with a reference equals logL without one, for
    real NG15 pulsars and arbitrary live params -- the decomposition is exact, and
    the frozen white-noise + reference logdets make even the absolute value match;
  * **reference-independent:** the value does not depend on which theta_ref is frozen;
  * **guardrail:** with reference=None the path is byte-identical to today (the live
    graph routes to the non-refdelta kernels).

The decisive HD-scale f32-vs-f64 accuracy table is a harness follow-up (the HD
analogue of finding_refdelta_table.md), not asserted here. See
piece2_fused_refdelta_plan.md (rung 3) and docs/adr/0001,0003.
"""
import contextlib
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

import discovery as ds
import discovery.recipes as R
from discovery import utils

DATA = Path(__file__).resolve().parents[2] / "data"
ARRAY_FEATHERS = [
    "v1p1_de440_pint_bipm2019-B1855+09.feather",
    "v1p1_de440_pint_bipm2019-B1937+21.feather",
    "v1p1_de440_pint_bipm2019-J1909-3744.feather",
]

# CURN single-level + fused HD: the two paths that gain a refdelta twin.
RECIPES = [
    pytest.param(R.intrinsic_plus_crn, id="curn_single_level"),
    pytest.param(R.intrinsic_rn_plus_global_hd, id="fused_hd"),
]


@pytest.fixture(scope="module")
def psrs():
    return [ds.Pulsar.read_feather(DATA / f) for f in ARRAY_FEATHERS]


@contextlib.contextmanager
def _metamath(dtype=jnp.float64):
    ds.config(kernels="metamath")
    utils.config(backend="jax", working=dtype)
    try:
        yield
    finally:
        utils.config(backend="jax")
        ds.config(kernels="matrix")


def _draw(params, seed):
    np.random.seed(seed)
    return ds.sample_uniform(params)


@pytest.mark.parametrize("recipe", RECIPES)
def test_reference_matches_baseline_f64(psrs, recipe):
    """logL with reference == logL without reference (f64), arbitrary live params."""
    with _metamath():
        base = recipe(psrs)
        params = base.logL.params
        theta_ref = _draw(params, 0)
        for seed in (1, 2, 3):
            theta = _draw(params, seed)
            L_base = float(base.logL(theta))
            rd = recipe(psrs)
            rd.reference = theta_ref
            L_rd = float(rd.logL(theta))
            np.testing.assert_allclose(L_rd, L_base, rtol=0, atol=1e-4,
                                       err_msg=f"{recipe.__name__} seed={seed}")


@pytest.mark.parametrize("recipe", RECIPES)
def test_reference_independent_f64(psrs, recipe):
    """The logL does not depend on which theta_ref is frozen."""
    with _metamath():
        base = recipe(psrs)
        params = base.logL.params
        theta = _draw(params, 7)
        vals = []
        for seed in (0, 10, 20):
            rd = recipe(psrs)
            rd.reference = _draw(params, seed)
            vals.append(float(rd.logL(theta)))
    np.testing.assert_allclose(vals, vals[0], rtol=0, atol=1e-4)


@pytest.mark.parametrize("recipe", RECIPES)
def test_optin_gate(psrs, recipe):
    """Guardrail: the refdelta opt-in is gated entirely by the frozen
    P_ref leaves. reference=None -> no P_ref on any kernel (today's graph,
    byte-identical); reference set -> a frozen P_ref appears on each live level."""
    from discovery import metamath
    with _metamath():
        # reference=None: opt-in OFF on every kernel.
        m = recipe(psrs)
        m.logL  # build the kernels
        # the marginal kernel assembly is the ground truth; ArrayLikelihood no
        # longer publishes it as a `self.vsm` side effect.
        assert getattr(m._marginal_assembly[0], "P_ref", None) is None
        if getattr(m, "gsm", None) is not None:
            assert getattr(m.gsm, "P_ref", None) is None

        # reference set: opt-in ON, frozen f64 covariance leaves. Assigning
        # `reference` invalidates the cached assemblies, so the rebuild below
        # sees it.
        rd = recipe(psrs)
        rd.reference = _draw(rd.logL.params, 0)
        rd.logL
        assert isinstance(rd._marginal_assembly[0].P_ref, metamath.NoiseMatrix)
        if getattr(rd, "gsm", None) is not None:
            assert isinstance(rd.gsm.P_ref, metamath.NoiseMatrix)
