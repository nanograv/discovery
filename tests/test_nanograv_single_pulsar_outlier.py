#!/usr/bin/env python3
"""Tests for discovery.models.nanograv_single_pulsar_outlier"""

import pathlib
import re

import pytest

# numpyro is an optional dependency: skip this whole module (rather than erroring
# at collection) when it is not installed, e.g. on CI where only base deps exist.
pytest.importorskip("numpyro")

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro

import discovery as ds
from discovery import prior
from discovery.models.nanograv_single_pulsar_outlier import (
    priordict_outlier_default,
    make_outlier_likelihood,
    _partition_params,
    assemble_pardict,
    draw_theta,
    draw_alpha,
    draw_z,
    draw_coeffs,
    make_outlier_gibbs_fn,
    make_outlier_model,
    run_outlier_mcmc,
    OutlierFitResult,
)


# Real-data fixture: J0437 feather shipped with discovery.
J0437_FEATHER = (pathlib.Path(ds.__path__[0]) / ".." / ".." / "data" /
                 "v1p1_de440_pint_bipm2019-J0437-4715.feather").resolve()


@pytest.fixture(scope="module", autouse=True)
def _matrix_mode():
    """The outlier model (`discovery.models.nanograv_single_pulsar_outlier`)
    builds on `likelihood.PulsarLikelihood` directly and reaches into matrix-only
    kernel APIs (`psrl.N.F`, `psrl.N.index`, `make_kernelsolve_simple`). The
    `signals` factory must therefore produce matrix kernels, which it only does
    under matrix mode -- no longer the default after the metamath default flip.

    Module-scoped and autouse so it runs BEFORE the module-scoped fixtures that
    build the likelihood (a function-scoped fixture would run too late). Saves
    and restores the module default so no state leaks to later test modules.
    Porting this analysis model to the graph path is a separate task."""
    saved = ds.config()
    ds.config(kernels="matrix")
    yield
    ds.config(kernels=saved)


@pytest.fixture(scope="module")
def j0437_psr():
    """J0437 Pulsar from the in-package data dir; skip cleanly if absent.

    The resulting object carries `.noisedict` so tests don't need a
    separate noisedict file.
    """
    if not J0437_FEATHER.exists():
        pytest.skip(f"missing fixture file: {J0437_FEATHER}")
    return ds.Pulsar.read_feather(str(J0437_FEATHER))


class _FakeLogL:
    def __init__(self, params):
        self.params = params


class _FakePsrl:
    def __init__(self, params):
        self.logL = _FakeLogL(params)


# ---- priordict_outlier_default ----

def test_priordict_extends_standard_with_outlier_entries():
    assert priordict_outlier_default["nu"] == [1, 40]
    assert priordict_outlier_default["theta_m"] == 0.01
    for k in prior.priordict_standard:
        assert k in priordict_outlier_default


# ---- _partition_params ----

def test_partition_params_buckets_correctly():
    params = [
        "J0437_1.5GHz_efac",
        "J0437_3GHz_efac",
        "J0437_1.5GHz_log10_t2equad",
        "J0437_1.5GHz_log10_tnequad",
        "J0437_1.5GHz_log10_ecorr",
        "J0437_red_noise_log10_rho(30)",
        "J0437_alpha_scaling(100)",
    ]
    parts = _partition_params(_FakePsrl(params))
    assert parts["efac"] == ["J0437_1.5GHz_efac", "J0437_3GHz_efac"]
    assert parts["equad"] == [
        "J0437_1.5GHz_log10_t2equad",
        "J0437_1.5GHz_log10_tnequad",
    ]
    assert parts["ecorr"] == ["J0437_1.5GHz_log10_ecorr"]
    assert parts["red_noise"] == ["J0437_red_noise_log10_rho(30)"]
    assert parts["alpha_scaling"] == "J0437_alpha_scaling(100)"


def test_partition_params_handles_powerlaw_red_noise():
    params = [
        "J0437_red_noise_log10_A",
        "J0437_red_noise_gamma",
        "J0437_alpha_scaling(100)",
    ]
    parts = _partition_params(_FakePsrl(params))
    assert parts["red_noise"] == [
        "J0437_red_noise_gamma",
        "J0437_red_noise_log10_A",
    ]


def test_partition_params_raises_without_alpha_scaling():
    with pytest.raises(ValueError, match="alpha_scaling"):
        _partition_params(_FakePsrl(["J0437_1.5GHz_efac"]))


def test_partition_params_raises_with_multiple_alpha_scaling():
    params = ["J0437_alpha_scaling(100)", "J1234_alpha_scaling(50)"]
    with pytest.raises(ValueError, match="alpha_scaling"):
        _partition_params(_FakePsrl(params))


# ---- assemble_pardict ----

def test_assemble_pardict_expands_array_sites():
    partition = {
        "efac":  ["J0437_1.5GHz_efac", "J0437_3GHz_efac"],
        "equad": ["J0437_1.5GHz_log10_t2equad"],
        "ecorr": ["J0437_1.5GHz_log10_ecorr"],
        "red_noise":    ["J0437_red_noise_log10_rho(30)"],
        "alpha_scaling": "J0437_alpha_scaling(100)",
    }
    hmc_sites = {
        "efacs":  jnp.array([1.0, 2.0]),
        "equads": jnp.array([-7.0]),
        "ecorrs": jnp.array([-7.5]),
        "nu":     jnp.array(5.0),
        "J0437_red_noise_log10_rho(30)": jnp.full(30, -7.0),
    }
    pardict = assemble_pardict(hmc_sites, partition)

    assert pardict["J0437_1.5GHz_efac"] == 1.0
    assert pardict["J0437_3GHz_efac"] == 2.0
    assert pardict["J0437_1.5GHz_log10_t2equad"] == -7.0
    assert pardict["J0437_1.5GHz_log10_ecorr"] == -7.5
    assert pardict["nu"] == 5.0
    assert pardict["J0437_red_noise_log10_rho(30)"].shape == (30,)


def test_assemble_pardict_handles_empty_partitions():
    # e.g. a pulsar with no ecorr or no red noise
    partition = {
        "efac":  ["J0437_efac"],
        "equad": [],
        "ecorr": [],
        "red_noise":    [],
        "alpha_scaling": "J0437_alpha_scaling(100)",
    }
    hmc_sites = {
        "efacs":  jnp.array([1.5]),
        "equads": jnp.array([]),
        "ecorrs": jnp.array([]),
    }
    pardict = assemble_pardict(hmc_sites, partition)
    assert pardict["J0437_efac"] == 1.5


# ---- make_outlier_likelihood (smoke tests against J0437) ----

def test_make_outlier_likelihood_builds_on_j0437(j0437_psr):
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    parts = _partition_params(psrl)
    # at least one of each WN class and exactly one alpha_scaling
    assert parts["efac"]
    assert parts["equad"]
    assert parts["ecorr"]
    assert parts["red_noise"]
    assert "alpha_scaling" in parts["alpha_scaling"]


def test_make_outlier_likelihood_logL_is_finite(j0437_psr):
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    # alpha_scaling has no prior pattern (it's Gibbs-sampled); fill ones manually
    parts = _partition_params(psrl)
    params = ds.sample_uniform(psrl.logL.params, fail=False)
    params[parts["alpha_scaling"]] = np.ones(j0437_psr.residuals.size)
    params = {**params, **j0437_psr.noisedict}
    val = float(psrl.logL(params))
    assert np.isfinite(val)


def test_make_outlier_likelihood_respects_components(j0437_psr):
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25,
                                   components=12)
    # red-noise param name carries the component count in parens
    rn = _partition_params(psrl)["red_noise"]
    assert any("(12)" in p for p in rn)


def test_make_outlier_likelihood_default_Tspan_uses_psr_span(j0437_psr):
    # smoke: no Tspan -> uses signals.getspan(psr), should still build
    psrl = make_outlier_likelihood(j0437_psr)
    parts = _partition_params(psrl)
    assert parts["red_noise"]


# ---- draw_theta ----

def test_draw_theta_posterior_mean_matches_conjugate():
    # With Beta(a, b) posterior, a/(a+b) is the analytic mean.
    N, k, m, z_sum = 1000, 1000, 0.01, 50
    z_i = jnp.zeros(N, dtype=jnp.int32).at[:z_sum].set(1)
    a = k * m + z_sum
    b = k * (1 - m) + N - z_sum
    expected_mean = a / (a + b)
    expected_var  = a * b / ((a + b) ** 2 * (a + b + 1))

    n_draws = 10000
    keys = jax.random.split(jax.random.key(0), n_draws)
    draws = jax.vmap(lambda rk: draw_theta(rk, z_i, k, m=m))(keys)
    mc_err = float(jnp.sqrt(expected_var / n_draws))

    assert abs(float(jnp.mean(draws)) - expected_mean) < 5 * mc_err


def test_draw_theta_clipping_floor_fires():
    # With sum(z)=0 and k=N=10, m=0.01, posterior is Beta(0.1, 19.9), heavily
    # concentrated near 0; eps=0.1 forces most draws to hit the lower clip.
    N, k, eps = 10, 10, 0.1
    z_i = jnp.zeros(N, dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(1), 500)
    draws = jax.vmap(lambda rk: draw_theta(rk, z_i, k, eps=eps))(keys)

    assert float(jnp.min(draws)) >= eps
    assert int(jnp.sum(draws == eps)) > 0


def test_draw_theta_shape_dtype_and_jits():
    fn = jax.jit(draw_theta)
    z_i = jnp.zeros(10, dtype=jnp.int32)
    theta = fn(jax.random.key(0), z_i, 10)
    assert theta.shape == ()
    assert bool(jnp.isfinite(theta))


# ---- draw_alpha ----

def test_draw_alpha_marginal_matches_inverse_gamma():
    # With z=1 everywhere, yp=1, sigma2=1: shape=(nu+1)/2, rate=(nu+1)/2,
    # so alpha_i ~ InverseGamma(3, 3): mean = rate/(shape-1) = 1.5,
    # var = rate^2 / ((shape-1)^2 (shape-2)) = 2.25.
    nu, N = 5.0, 10
    z_i = jnp.ones(N, dtype=jnp.int32)
    yprime = jnp.ones(N)
    sigma2 = jnp.ones(N)
    expected_mean, expected_var = 1.5, 2.25

    n_draws = 10000
    keys = jax.random.split(jax.random.key(2), n_draws)
    draws = jax.vmap(lambda rk: draw_alpha(rk, z_i, nu, yprime, sigma2))(keys)
    # Each TOA is iid, so pool all (n_draws * N) samples for the mean estimate.
    sample_mean = float(jnp.mean(draws))
    mc_err = float(jnp.sqrt(expected_var / (n_draws * N)))
    assert abs(sample_mean - expected_mean) < 5 * mc_err


def test_draw_alpha_z_zero_independent_of_yprime():
    # When z=0, the rate term `z * (yp^2 / sigma^2)` vanishes; with the same
    # PRNG key the draw is identical regardless of the residual magnitude.
    N, nu = 100, 5.0
    z_i = jnp.zeros(N, dtype=jnp.int32)
    sigma2 = jnp.ones(N)
    key = jax.random.key(3)

    a1 = draw_alpha(key, z_i, nu, jnp.zeros(N),         sigma2)
    a2 = draw_alpha(key, z_i, nu, jnp.full(N, 1e6),     sigma2)
    assert jnp.allclose(a1, a2)


def test_draw_alpha_shape_dtype_and_jits():
    fn = jax.jit(draw_alpha)
    N = 50
    z_i = jnp.ones(N, dtype=jnp.int32)
    alpha = fn(jax.random.key(0), z_i, 5.0, jnp.zeros(N), jnp.ones(N))
    assert alpha.shape == (N,)
    assert bool(jnp.all(jnp.isfinite(alpha)))
    assert bool(jnp.all(alpha > 0))


# ---- draw_z ----

def test_draw_z_theta_near_zero_yields_all_zero():
    # With theta -> 0, q -> 0 regardless of the data; z_i should be all 0.
    N = 100
    yprime = jnp.zeros(N)
    s2 = jnp.ones(N)
    z, q = draw_z(jax.random.key(0), 1e-10, yprime, 4 * s2, s2)
    assert float(jnp.max(q)) < 1e-6
    assert int(jnp.sum(z)) == 0


def test_draw_z_theta_near_one_yields_all_one():
    N = 100
    yprime = jnp.zeros(N)
    s2 = jnp.ones(N)
    z, q = draw_z(jax.random.key(0), 1.0 - 1e-10, yprime, 4 * s2, s2)
    assert float(jnp.min(q)) > 1.0 - 1e-6
    assert int(jnp.sum(z)) == N


def test_draw_z_analytic_q():
    # With yprime = 0 and theta = 0.5, sigma2_scaled = 4 * sigma2_unit:
    #   p(z=1) ∝ 0.5 * N(0|0, 4) = 0.5 / sqrt(8 pi)
    #   p(z=0) ∝ 0.5 * N(0|0, 1) = 0.5 / sqrt(2 pi)
    # so q = (1/sqrt(4)) / (1/sqrt(4) + 1) = 1/3.
    yprime  = jnp.zeros(5)
    s2_unit = jnp.ones(5)
    s2_sc   = 4 * s2_unit
    _, q = draw_z(jax.random.key(0), 0.5, yprime, s2_sc, s2_unit)
    assert jnp.allclose(q, 1.0 / 3.0, atol=1e-12)


def test_draw_z_bernoulli_stats_match_q():
    # Self-consistent: the deterministic q from the inputs is 1/3 (see above),
    # so the empirical Bernoulli mean across many draws should match within MC.
    N = 50
    yprime  = jnp.zeros(N)
    s2_unit = jnp.ones(N)
    s2_sc   = 4 * s2_unit

    n_draws = 5000
    keys = jax.random.split(jax.random.key(0), n_draws)
    zs, _ = jax.vmap(lambda rk: draw_z(rk, 0.5, yprime, s2_sc, s2_unit))(keys)
    q = 1.0 / 3.0
    mc_err = float(jnp.sqrt(q * (1 - q) / (n_draws * N)))
    assert abs(float(jnp.mean(zs)) - q) < 5 * mc_err


def test_draw_z_shape_dtype_and_jits():
    fn = jax.jit(draw_z)
    N = 30
    yprime = jnp.zeros(N)
    s2 = jnp.ones(N)
    z, q = fn(jax.random.key(0), 0.5, yprime, 4 * s2, s2)
    assert z.shape == (N,)
    assert q.shape == (N,)
    assert z.dtype == jnp.int32
    assert bool(jnp.all((z == 0) | (z == 1)))
    assert bool(jnp.all((q >= 0) & (q <= 1)))


# ---- draw_coeffs ----

def test_draw_coeffs_flattens_in_cvars_order():
    # Stub `sample_cond_fn` returns a fixed dict; verify hstack ordering.
    def stub(key, _params):
        return key, {
            "a": jnp.array([1.0, 2.0]),
            "b": jnp.array([3.0]),
            "c": jnp.array([4.0, 5.0]),
        }

    cdict, cflat = draw_coeffs(jax.random.key(0), {}, stub, ["a", "b", "c"])
    assert jnp.allclose(cflat, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    _, cflat_rev = draw_coeffs(jax.random.key(0), {}, stub, ["c", "a", "b"])
    assert jnp.allclose(cflat_rev, jnp.array([4.0, 5.0, 1.0, 2.0, 3.0]))

    # underlying dict is forwarded unchanged
    assert set(cdict.keys()) == {"a", "b", "c"}


# ---- make_outlier_model ----

def test_outlier_model_registers_expected_sites(j0437_psr):
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    model = make_outlier_model(psrl)
    parts = _partition_params(psrl)
    N = j0437_psr.residuals.size
    n_coeffs = psrl.N.F.shape[-1]

    seeded = numpyro.handlers.seed(model, rng_seed=0)
    trace  = numpyro.handlers.trace(seeded).get_trace()

    # HMC sites
    assert trace["efacs"]["value"].shape  == (len(parts["efac"]),)
    assert trace["equads"]["value"].shape == (len(parts["equad"]),)
    assert trace["ecorrs"]["value"].shape == (len(parts["ecorr"]),)
    assert trace["nu"]["value"].shape     == ()
    for rn_name in parts["red_noise"]:
        assert rn_name in trace

    # Gibbs sites (placeholder priors; values discarded by the gibbs_fn)
    assert trace["coeffs"]["value"].shape  == (n_coeffs,)
    assert trace["theta"]["value"].shape   == ()
    assert trace["z_i"]["value"].shape     == (N,)
    assert trace["q"]["value"].shape       == (N,)
    assert trace["alpha_i"]["value"].shape == (N,)

    # Determinstics
    assert np.isfinite(float(trace["loglike"]["value"]))
    assert isinstance(trace["params"]["value"], dict)


def test_outlier_model_priors_pulled_from_priordict(j0437_psr):
    # Override the priordict and check the drawn values lie in the overridden
    # ranges (Uniform draws are guaranteed to be in [low, high]).
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    custom = {
        **priordict_outlier_default,
        "(.*_)?efac": [0.5, 1.5],
        "nu": [2, 10],
    }
    model = make_outlier_model(psrl, priordict=custom)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, rng_seed=0)).get_trace()

    efacs = trace["efacs"]["value"]
    nu    = float(trace["nu"]["value"])
    assert bool(jnp.all((efacs >= 0.5) & (efacs <= 1.5)))
    assert 2 <= nu <= 10


def test_outlier_model_handles_powerlaw_psd(j0437_psr):
    # With psd=powerlaw the red-noise hypers are scalar log10_A + gamma
    # (no parenthesized count); the model should still build cleanly.
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25,
                                   psd=ds.signals.powerlaw)
    model = make_outlier_model(psrl)
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, rng_seed=0)).get_trace()

    rn_sites = [s for s in trace if "red_noise" in s and "coefficients" not in s]
    assert any("log10_A" in s for s in rn_sites)
    assert any("gamma"   in s for s in rn_sites)
    assert np.isfinite(float(trace["loglike"]["value"]))


# ---- orchestrator: make_outlier_gibbs_fn ----

@pytest.fixture(scope="module")
def j0437_gibbs_state(j0437_psr):
    """Initial (psrl, gibbs_fn, gibbs_sites, hmc_sites) for J0437.

    `hmc_sites` is seeded from `psr.noisedict` for WN, mid-prior for RN /
    nu, so `gibbs_fn` sees a physically reasonable starting state.
    """
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    gibbs_fn = make_outlier_gibbs_fn(psrl)
    parts = _partition_params(psrl)
    nd = j0437_psr.noisedict
    N = j0437_psr.residuals.size
    n_coeffs = psrl.N.F.shape[-1]

    gibbs_sites = {
        "coeffs":  jnp.zeros(n_coeffs),       # overwritten on first call
        "theta":   jnp.array(0.05),
        "z_i":     jnp.zeros(N, dtype=jnp.int32),
        "alpha_i": jnp.ones(N),
        "q":       jnp.zeros(N),
    }
    hmc_sites = {
        "efacs":  jnp.array([nd[p] for p in parts["efac"]]),
        "equads": jnp.array([nd[p] for p in parts["equad"]]),
        "ecorrs": jnp.array([nd[p] for p in parts["ecorr"]]),
        "nu":     jnp.array(5.0),
    }
    for rn_param in parts["red_noise"]:
        # red-noise sites carry their length in their name, e.g. "...log10_rho(30)"
        m = re.search(r"\((\d+)\)$", rn_param)
        size = int(m.group(1)) if m else 1
        hmc_sites[rn_param] = jnp.full(size, -7.0)

    return psrl, gibbs_fn, gibbs_sites, hmc_sites


def test_gibbs_fn_smoke_on_j0437(j0437_gibbs_state):
    _, gibbs_fn, gs, hs = j0437_gibbs_state
    out = gibbs_fn(jax.random.key(0), gs, hs)
    for site in ["coeffs", "theta", "z_i", "alpha_i", "q"]:
        assert site in out
        assert bool(jnp.all(jnp.isfinite(out[site]))), f"NaN at site {site}"


def test_gibbs_fn_finite_repeated_under_jit(j0437_gibbs_state):
    # Full-orchestrator analogue of the draw_coeffs regression test:
    # 10 jitted calls with different keys, all sites finite.
    _, gibbs_fn, gs, hs = j0437_gibbs_state
    jgibbs = jax.jit(gibbs_fn)
    for i in range(10):
        out = jgibbs(jax.random.key(i), gs, hs)
        for site in ["coeffs", "theta", "z_i", "alpha_i", "q"]:
            assert bool(jnp.all(jnp.isfinite(out[site]))), \
                f"NaN at iter {i}, site {site}"


def test_gibbs_fn_deterministic_jit_vs_eager(j0437_gibbs_state):
    # Same key + same state must produce identical output, eager and JIT.
    _, gibbs_fn, gs, hs = j0437_gibbs_state
    key = jax.random.key(42)

    out_eager_1 = gibbs_fn(key, gs, hs)
    out_eager_2 = gibbs_fn(key, gs, hs)
    out_jit     = jax.jit(gibbs_fn)(key, gs, hs)

    for site in ["coeffs", "theta", "alpha_i", "q"]:
        assert jnp.allclose(out_eager_1[site], out_eager_2[site], atol=1e-12), \
            f"eager non-deterministic at {site}"
        assert jnp.allclose(out_eager_1[site], out_jit[site], atol=1e-10), \
            f"jit mismatch at {site}"
    assert jnp.array_equal(out_eager_1["z_i"], out_eager_2["z_i"])
    assert jnp.array_equal(out_eager_1["z_i"], out_jit["z_i"])


# ---- run_outlier_mcmc end-to-end ----

@pytest.fixture(scope="module")
def j0437_mcmc_result(j0437_psr):
    """Run a tiny MCMC end-to-end on J0437, reused by all integration tests.

    `num_samples=10` is the minimum that lets `numpyro.diagnostics.summary`
    compute split-Rhat (requires `num_samples // 2 >= 4`).
    """
    return run_outlier_mcmc(
        j0437_psr,
        rng_key=jax.random.key(0),
        num_warmup=2,
        num_samples=10,
        max_tree_depth=1,
        Tspan=20 * 86400 * 365.25,
    )


def test_run_outlier_mcmc_returns_result(j0437_mcmc_result):
    assert isinstance(j0437_mcmc_result, OutlierFitResult)
    s = j0437_mcmc_result.samples
    for key in ["efacs", "equads", "ecorrs", "nu",
                "theta", "z_i", "alpha_i", "q", "coeffs",
                "loglike", "params"]:
        assert key in s, f"missing sample key {key}"


def test_outlier_fit_result_outlier_mask(j0437_mcmc_result):
    mask = j0437_mcmc_result.outlier_mask(threshold=0.0)
    assert mask.shape == (j0437_mcmc_result.psr.residuals.size,)
    assert mask.dtype == bool


def test_outlier_fit_result_whitened_residuals(j0437_mcmc_result):
    wr = j0437_mcmc_result.whitened_residuals(sample_idx=0)
    assert wr.shape == (j0437_mcmc_result.psr.residuals.size,)
    assert np.all(np.isfinite(wr))


def test_outlier_fit_result_summary_and_rhat(j0437_mcmc_result):
    summary = j0437_mcmc_result.summary()
    assert "efacs" in summary
    # rhat returns a dict when called with no arg
    rhats = j0437_mcmc_result.rhat()
    assert isinstance(rhats, dict)
    assert "nu" in rhats
    # single-site form
    nu_rhat = j0437_mcmc_result.rhat("nu")
    assert np.isscalar(nu_rhat) or hasattr(nu_rhat, "shape")


def test_draw_coeffs_finite_under_jit_j0437(j0437_psr):
    # Regression guard for the cho_factor double-factor bug: under JIT,
    # `psrl.sample_conditional` was producing intermittent NaNs (~50% rate).
    # We now expect all 10 jitted calls to be finite.
    psrl = make_outlier_likelihood(j0437_psr, Tspan=20 * 86400 * 365.25)
    parts = _partition_params(psrl)
    params = ds.sample_uniform(psrl.logL.params, fail=False)
    params[parts["alpha_scaling"]] = np.ones(j0437_psr.residuals.size)
    params = {**params, **j0437_psr.noisedict}

    cvars = list(psrl.N.index.keys())
    sample_cond = psrl.sample_conditional

    @jax.jit
    def jcall(key, pd):
        return draw_coeffs(key, pd, sample_cond, cvars)

    for i in range(10):
        _, cflat = jcall(jax.random.key(i), params)
        assert bool(jnp.all(jnp.isfinite(cflat))), f"NaN at iteration {i}"


def test_assemble_pardict_works_under_jit():
    # Smoke test: the partition is static, only the hmc_sites values are
    # tracers. JIT should compile and run.
    partition = {
        "efac":  ["a", "b"],
        "equad": ["c"],
        "ecorr": ["d"],
        "red_noise":    [],
        "alpha_scaling": "alpha",
    }

    @jax.jit
    def f(hmc_sites):
        pd = assemble_pardict(hmc_sites, partition)
        return pd["a"] + pd["b"] + pd["c"] + pd["d"]

    hmc_sites = {
        "efacs":  jnp.array([1.0, 2.0]),
        "equads": jnp.array([3.0]),
        "ecorrs": jnp.array([4.0]),
    }
    assert float(f(hmc_sites)) == 10.0
