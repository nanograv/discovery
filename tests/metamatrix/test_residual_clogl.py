"""Residual-form coefficient likelihood.

`ArrayLikelihood.clogL` has two algebraically identical forms:

  * "cross"    — `vectorgpcomponent`, which forms F^T N^-1 F per pulsar;
  * "residual" — `vectorresidualcomponent`, which never pushes anything of
                 shape (n_toa, k) through a noise solve.

With constant N the cross form's products fold at trace time and cost nothing
per evaluation. With parameter-dependent N they are rebuilt at every evaluation,
O(n_toa * k^2) — which is what the residual form exists to avoid, and why
`clogl_form="auto"` routes to it exactly when a per-pulsar noise solve has free
parameters.
"""
import time

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
from discovery import metamatrix as mm  # noqa: E402


SEED = 20260716


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


# --------------------------------------------------------------------------
# model builders
# --------------------------------------------------------------------------

def _psl(psr, fixed_wn=True):
    return ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict if fixed_wn else {}),
        ds.makegp_timing(psr, svd=True),
    ])


def _model(psrs, form, *, fixed_wn=True, components=30, means=None,
           extsignals=None, reference=None):
    T = ds.getspan(psrs)
    return ds.ArrayLikelihood(
        [_psl(p, fixed_wn) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=components,
                                         T=T, name="rednoise", means=means),
        extsignals=extsignals, reference=reference, clogl_form=form)


def _draws(params, n):
    """`n` parameter draws. Immediately before the batch, seed NumPy's legacy
    global RNG — that is the contract `prior.sample_uniform` still follows.
    Coefficient keys (and any hyperparameter with no standard prior, e.g. a
    non-zero-mean amplitude) are drawn from the same seeded global RNG."""
    def has_prior(par):
        try:
            ds.getprior_uniform(par)
            return True
        except Exception:
            return False

    coeff = [p for p in params if "_coefficients(" in p]
    hyper = [p for p in params if p not in coeff and has_prior(p)]
    other = [p for p in params if p not in coeff and p not in hyper]

    np.random.seed(SEED)
    out = []
    for _ in range(n):
        p0 = ds.sample_uniform(hyper)
        for key in coeff:
            width = int(key[key.index("(") + 1:key.index(")")])
            p0[key] = np.random.randn(width)
        for key in other:
            p0[key] = float(np.random.randn())
        out.append(p0)
    return out


def _value(model, p0):
    out = model.clogL(p0)
    return float(out[0]) if isinstance(out, tuple) else float(out)


def _assert_forms_agree(psrs, rtol=1e-10, n=20, **kw):
    cross = _model(psrs, "cross", **kw)
    resid = _model(psrs, "residual", **kw)

    p0s = _draws(sorted(cross.clogL.params), n)
    a = [_value(cross, p0) for p0 in p0s]
    b = [_value(resid, p0) for p0 in p0s]

    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(b, a, rtol=rtol)
    return a, b


# --------------------------------------------------------------------------
# 1-3: the two forms are the same function
# --------------------------------------------------------------------------

def test_form_equivalence_constant_noise(psrs, metamath_backend):
    _assert_forms_agree(psrs)


def test_form_equivalence_with_extsignal(psrs, metamath_backend):
    """Certifies the subtraction-vs-cross-term identity: the residual form
    subtracts each ExtSignal from r, where the cross form expands the
    cross-terms explicitly."""
    cw = ds.makecw_extsignal(psrs, components=50, T=ds.getspan(psrs), name="cw")
    _assert_forms_agree(psrs, extsignals=[cw])


def test_form_equivalence_with_means(psrs, metamath_backend):
    """The prior centering flows through `_coefficient_leaves` identically."""
    def my_means(f, df, mean_amp):
        return mean_amp * jax.numpy.ones_like(f)

    _assert_forms_agree(psrs, means=my_means)


# --------------------------------------------------------------------------
# 4: varying white noise, against an independent dense oracle
# --------------------------------------------------------------------------

def test_varying_noise_against_a_dense_numpy_oracle(psrs, metamath_backend):
    """The residual form's reason to exist. Checked against an explicit dense
    N(theta) build and residual quadratic — an oracle that shares no code with
    the graph path.

    The per-pulsar kernel is bare measurement noise (no marginalized timing GP),
    so N(theta) is exactly diagonal and the oracle stays numerically clean; the
    varying-white-noise fast path is what is under test regardless.
    """
    T = ds.getspan(psrs)
    model = ds.ArrayLikelihood(
        [ds.PulsarLikelihood([p.residuals, ds.makenoise_measurement(p, {})])
         for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=10, T=T,
                                         name="rednoise"),
        clogl_form="residual")

    assert model.clogl_form_resolved == "residual"
    p0 = _draws(sorted(model.clogL.params), 1)[0]
    got = _value(model, p0)

    vsm, ys = model._coefficient_assembly
    index_per_psr = (vsm.index if isinstance(vsm.index, list)
                     else [{k: v} for k, v in vsm.index.items()])

    # --- oracle: log p(y, c) = sum_i [-0.5 r^T N^-1 r - 0.5 log|N|] + log p(c)
    expected = 0.0
    for i, psr in enumerate(psrs):
        # N(theta): per-backend efac^2 * (toaerrs^2 + 10^(2 log10_t2equad))
        flags = ds.selection_backend_flags(psr)
        backends = [b for b in sorted(set(flags)) if b != ""]
        N = np.zeros(len(psr.toas))
        for b in backends:
            mask = (flags == b)
            efac = p0[f"{psr.name}_{b}_efac"]
            equad2 = 10.0 ** (2 * p0[f"{psr.name}_{b}_log10_t2equad"])
            N += mask * efac ** 2 * (psr.toaerrs ** 2 + equad2)

        c_i = np.concatenate([p0[k] for k in index_per_psr[i]])
        r = np.asarray(ys[i]) - np.asarray(vsm.Fs[i]) @ c_i

        expected += -0.5 * np.sum(r ** 2 / N) - 0.5 * np.sum(np.log(N))

    # log p(c) under the commongp's diagonal powerlaw prior
    phi = np.asarray(model.commongp.Phi.getN(p0))
    for i in range(len(psrs)):
        c_i = np.concatenate([p0[k] for k in index_per_psr[i]])
        expected += -0.5 * np.sum(c_i ** 2 / phi[i]) - 0.5 * np.sum(np.log(phi[i]))

    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_auto_resolves_to_residual_under_varying_noise(psrs, metamath_backend):
    assert _model(psrs, "auto", fixed_wn=False).clogl_form_resolved == "residual"


def test_auto_resolves_to_cross_under_constant_noise(psrs, metamath_backend):
    """With constant noise the cross form's products fold at trace time, so
    `auto` leaves the existing graph selected."""
    assert _model(psrs, "auto", fixed_wn=True).clogl_form_resolved == "cross"


def test_unknown_clogl_form_raises_at_construction(psrs, metamath_backend):
    with pytest.raises(ValueError, match="unknown clogl_form"):
        _model(psrs, "residaul")


def test_residual_form_requires_metamath(psrs):
    """clogl_form='residual' is a metamath-only entry point: the guard in
    the residual branch raises if the metamath path is not active when clogL is
    assembled. Build the metamath likelihood, then flip the mode away before
    touching clogL."""
    ds.config(kernels="metamath")
    try:
        model = _model(psrs, "residual")
        assert model.clogl_form_resolved == "residual"    # resolved while metamath
        ds.config(kernels="matrix")                        # now flip the mode
        with pytest.raises(NotImplementedError, match="metamath kernel path"):
            model.clogL
    finally:
        ds.config(kernels="matrix")


# --------------------------------------------------------------------------
# 5: the topology gate — no noise solve consumes a GP design matrix
# --------------------------------------------------------------------------

def _noise_solve_applications(graph):
    """[(callee, [args])] for every Apply on a `vectorresidualcomponent` /
    `vectorgpcomponent` noise-solve leaf, over `graph` and every nested graph.

    Scoped to the component graph's OWN `Nsolves_i` / `Fs_i` leaves. A nested
    per-pulsar Woodbury solve has its own local `Nsolve` / `F` leaves and
    legitimately applies one to the other — that is the inner block it
    marginalizes. The claim under test is about the OUTER GP basis.
    """
    found = []
    seen = set()

    def visit(g):
        if id(g) in seen:
            return
        seen.add(id(g))

        for name, node in g.items():
            if isinstance(node, mm.Node) and node.op is mm.Apply:
                callee = node.inputs[0]
                if callee not in g:
                    raise AssertionError(
                        f"unresolved Apply callee {callee!r} in graph")
                if not callee.startswith("Nsolves_"):
                    continue
                for arg in node.inputs[1:]:
                    if arg not in g:
                        raise AssertionError(
                            f"unresolved Apply input {arg!r} in graph")
                found.append((callee, list(node.inputs[1:])))
            elif isinstance(node, mm.GraphLeaf):
                visit(node.graph)
            elif isinstance(node, mm.FuncLeaf):
                nested = getattr(node.fn, "graph", None)
                if isinstance(nested, dict):
                    visit(nested)

    visit(graph)
    return found


def test_residual_graph_never_applies_a_noise_solve_to_a_gp_basis(psrs, metamath_backend):
    """The deterministic regression gate against accidental FtNmF
    materialization. Inspects the pruned graph BEFORE fold_constants /
    metamatrix.func, so nothing is hidden by folding."""
    model = _model(psrs, "residual", fixed_wn=False, components=10)
    vsm, ys = model._coefficient_assembly
    graph = vsm.make_residualproduct(ys)

    applications = _noise_solve_applications(graph)
    assert applications, "expected the residual graph to apply each noise solve"

    for callee, args in applications:
        for arg in args:
            assert not arg.startswith("Fs_"), (
                f"{callee} is applied directly to the GP basis {arg}: the "
                f"residual form must never push a design matrix through a "
                f"noise solve")
            assert isinstance(graph[arg], mm.Node), (
                f"{callee} should consume the residual expression (a Node); "
                f"got leaf {arg}")


def test_the_topology_gate_discriminates(psrs, metamath_backend):
    """The cross form DOES apply each noise solve to its GP basis. Without this,
    the assertion above could pass vacuously."""
    model = _model(psrs, "cross", fixed_wn=False, components=10)
    vsm, ys = model._coefficient_assembly
    graph = vsm.make_kernelproduct_gpcomponent(ys)

    args = [a for _, arglist in _noise_solve_applications(graph) for a in arglist]

    assert any(a.startswith("Fs_") for a in args)


# --------------------------------------------------------------------------
# 6: the exact Laplace identity for a Gaussian
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_gaussian_identity_against_the_marginal_likelihood(psrs, metamath_backend):
    """The exact Gaussian marginalization identity: integrating the sampled
    coefficients out of the residual-form `clogL` reproduces the marginal
    `logL`. commongp only: `ArrayLikelihood.conditional` does not support a
    global GP. Every block needs a PROPER prior, so the timing model goes in as
    a variable GP with a finite variance.

    Convention note: discovery's coefficient log-prior drops the (2*pi)^{-k/2}
    normalizer, so the (2*pi)^{k/2} factor from the Gaussian integral is exactly
    absorbed and does NOT appear. The identity is therefore

        logL == clogL(c_hat) - 0.5 log|A|

    with c_hat, A the conditional mean and precision. (Verified independent of
    the timing variance: with the 2*pi term added the mismatch is a constant
    0.5*k*log(2*pi), the tell-tale of the omitted normalizer.)
    """
    T = ds.getspan(psrs)
    psls = [ds.PulsarLikelihood([
        p.residuals,
        ds.makenoise_measurement(p, p.noisedict),
        ds.makegp_timing(p, svd=True, variable=True, variance=1e-12),
    ]) for p in psrs]

    def build(form):
        return ds.ArrayLikelihood(
            psls, commongp=ds.makecommongp_fourier(
                psrs, ds.powerlaw, components=10, T=T, name="rednoise"),
            clogl_form=form)

    model = build("residual")
    p0 = _draws(sorted(model.logL.params), 1)[0]

    mu, cf = model.conditional(p0)
    mu = np.asarray(mu)                       # (npsr, k)

    # log p(y, c_hat) at the conditional mean. The commongp coefficient index is
    # a flat dict, one key per pulsar in pulsar order; the conditional mean is a
    # per-pulsar (npsr, k) array, so each pulsar's key takes its whole row.
    p_hat = dict(p0)
    keys = list(model.conditional_index if hasattr(model, "conditional_index")
                else model._marginal_assembly[0].index)
    for i, key in enumerate(keys):
        p_hat[key] = mu[i]

    joint = _value(model, p_hat)

    # A = the conditional precision; cf is its batched (npsr, k, k) lower
    # Cholesky, so log|A| sums 2*log diag over all pulsars.
    L = np.asarray(cf[0])
    logdetA = 2.0 * np.sum(np.log(np.abs(np.diagonal(L, axis1=-2, axis2=-1))))

    laplace = joint - 0.5 * logdetA

    np.testing.assert_allclose(laplace, float(model.logL(p0)), rtol=1e-8)


# --------------------------------------------------------------------------
# 7: informational scaling benchmark (never gates)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_scaling_benchmark_is_informational_only(psr, metamath_backend, capsys):
    """Steady-state per-eval wall times for both forms under varying white
    noise, at two component counts so the k^2-vs-k scaling is visible.

    NO wall-clock ratio is asserted: hardware, JAX version, compilation cache
    and runner contention make such a gate flaky. Test 5's topology assertion is
    the deterministic regression gate; this supplies the performance evidence.
    """
    psrs = [psr]

    def timed(form, components):
        model = _model(psrs, form, fixed_wn=False, components=components)
        p0 = _draws(sorted(model.clogL.params), 1)[0]
        fn = jax.jit(model.clogL)

        out = fn(p0)                       # compile
        jax.block_until_ready(out)

        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            jax.block_until_ready(fn(p0))
        return (time.perf_counter() - t0) / reps

    rows = []
    for components in (15, 30):
        for form in ("cross", "residual"):
            rows.append((form, components, timed(form, components)))

    lines = ["", "clogL steady-state per-eval wall time (varying white noise)"]
    for form, components, dt in rows:
        lines.append(f"  {form:>8s}  k={2 * components:<3d}  {dt * 1e3:8.3f} ms")
    t = {(f, c): dt for f, c, dt in rows}
    lines.append(f"  ratio cross/residual at k=60: "
                 f"{t[('cross', 30)] / t[('residual', 30)]:.2f}x")
    lines.append(f"  cross    k=30 -> k=60 growth: "
                 f"{t[('cross', 30)] / t[('cross', 15)]:.2f}x")
    lines.append(f"  residual k=30 -> k=60 growth: "
                 f"{t[('residual', 30)] / t[('residual', 15)]:.2f}x")

    with capsys.disabled():
        print("\n".join(lines))

    assert all(dt > 0 for _, _, dt in rows)


# --------------------------------------------------------------------------
# 8: reference= coexistence
# --------------------------------------------------------------------------

def test_reference_and_residual_clogl_coexist(psrs, metamath_backend):
    """`reference=` routes the marginal path; `clogl_form=` routes the
    coefficient path. Both cached properties live on one instance and neither
    disturbs the other."""
    theta_ref = _draws(sorted(_model(psrs, "cross").logL.params), 1)[0]
    theta_ref = {k: v for k, v in theta_ref.items() if "_coefficients(" not in k}

    both = _model(psrs, "residual", reference=theta_ref)
    logl_twin = _model(psrs, "cross", reference=theta_ref)
    clogl_twin = _model(psrs, "residual")

    p0 = _draws(sorted(both.clogL.params), 1)[0]

    np.testing.assert_allclose(float(both.logL(p0)), float(logl_twin.logL(p0)),
                               rtol=1e-12)
    np.testing.assert_allclose(_value(both, p0), _value(clogl_twin, p0), rtol=1e-12)
