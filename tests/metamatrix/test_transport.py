"""Graph-consistent transport.

`Transport` / `ArrayTransport` replace the in-likelihood decenter closure with a
free-standing reparam object that declares its true `.params`, carries a
mandatory exact conditioner precision for diagonal priors (no floors, no
ridges), and has specified failure semantics at construction / validate /
runtime.
"""
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import discovery.recipes as R  # noqa: E402
from discovery import utils as kh  # noqa: E402
from discovery import metamatrix  # noqa: E402
from discovery import transport as tr  # noqa: E402


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


@pytest.fixture
def cholesky_backend(metamath_backend):
    """The transport parity anchor is the closure's DEFAULT Cholesky config."""
    ds.utils.config(backend="jax", factor="cholesky")
    assert ds.utils.matrix_norm == 2.0
    yield


# ==========================================================================
# 1. Closure parity — the deletion gate
# ==========================================================================

def _legacy_decenter_transform(model):
    """Standalone replica of the deleted decenter closure
    (likelihood_metamath.py at the pre-deletion commit;
    `legacy_decenter_transform`). Returns rp(params, c) -> (q, ldL). This is
    the golden source the transport is certified against."""
    vsm, ys = model._coefficient_assembly

    def _solve_2d(N, F):
        return metamatrix.func(N.make_solve)(F, params={})

    def _eval_F(F):
        if isinstance(F, dict):
            return kh.jnp.asarray(metamatrix.func(F)(params={}))
        return kh.jnp.asarray(F)

    vsm_Fs = [_eval_F(F) for F in vsm.Fs]
    NmFs, _ldNs = zip(*[_solve_2d(N, F) for N, F in zip(vsm.Ns, vsm_Fs)])
    FtNmFs = [F.T @ NmF for F, NmF in zip(vsm_Fs, NmFs)]
    NmFtys = [NmF.T @ y for NmF, y in zip(NmFs, ys)]
    FtNmF, NmFty = kh.jnparray(FtNmFs), kh.jnparray(NmFtys)

    def decenter_transform(params, c):
        cgp_list = (model.commongp if isinstance(model.commongp, list)
                    else [model.commongp])
        phis_invs_commongp = [gp.Phi.getN(params) ** -1 for gp in cgp_list]
        if model.globalgp is not None:
            phis_invs_globalgp = (kh.jnp.diag(
                model.globalgp.Phi.getN(params) ** -1
            ).reshape((len(model.psls), -1)))
            phis_invs = kh.jnp.concatenate(
                [*phis_invs_commongp, phis_invs_globalgp], axis=1)
        else:
            phis_invs = kh.jnp.concatenate([*phis_invs_commongp], axis=1)
        i1, i2 = kh.jnp.diag_indices(phis_invs.shape[1], ndim=2)
        cf = kh.matrix_factor(FtNmF.at[:, i1, i2].add(phis_invs), lower=True)
        am = kh.jsp.linalg.solve_triangular(cf[0], c, trans=1, lower=cf[1])
        mus = kh.matrix_solve(cf, NmFty)
        ldL = -kh.jnp.logdet(cf[0][:, i1, i2])
        return am + mus, ldL

    decenter_transform.params = []
    return decenter_transform


@pytest.mark.parametrize("recipe", ["decenter_intrinsic_rn",
                                    "decenter_intrinsic_rn_global_hd"])
def test_closure_parity_is_the_deletion_gate(psrs, cholesky_backend, recipe):
    """Transport-based reparam vs the closure replica at 20 draws.

    logp (the clogL value) and ldJ match to rtol=1e-12; the raw transformed
    coefficients match to rtol=1e-9. The looser coefficient tolerance is honest
    float64 behavior, not a defect: the transport bakes G0/b0 in NumPy while the
    closure builds them in JAX, and the two BLAS paths diverge at ~1e-11 on the
    ill-conditioned Cholesky solve. That divergence washes out of logp/ldJ,
    which is why the LIKELIHOOD matches to machine precision.
    """
    model = getattr(R, recipe)(psrs)
    legacy = _legacy_decenter_transform(model)
    transport = model._build_decenter_transport(model._coefficient_assembly[1])
    rp = transport.as_reparam()

    npsr = len(psrs)
    k = transport.dimension
    hyper = [p for p in sorted(model.clogL.params) if "_coefficients(" not in p]

    np.random.seed(20260716)
    for _ in range(20):
        params = ds.sample_uniform(hyper)
        xi = kh.jnparray(np.random.randn(npsr, k))

        q_old, ldj_old = legacy(params, xi)
        q_new, ldj_new = rp(params, xi)

        # Compare coefficients relative to their overall scale, not per element:
        # near-zero entries carry no meaningful relative tolerance.
        qo, qn = np.asarray(q_old), np.asarray(q_new)
        scale = np.max(np.abs(qo))
        assert np.max(np.abs(qn - qo)) <= 1e-9 * scale
        np.testing.assert_allclose(float(ldj_new), float(ldj_old), rtol=1e-12)


@pytest.mark.parametrize("recipe", ["decenter_intrinsic_rn",
                                    "decenter_intrinsic_rn_global_hd"])
def test_transport_clogl_matches_the_closure_likelihood(psrs, cholesky_backend, recipe):
    """The clogL VALUE under the transport equals the closure's, to rtol=1e-12
    (in practice ~1e-15). This is the parity that actually gates deletion."""
    model = getattr(R, recipe)(psrs)

    vsm, ys = model._coefficient_assembly
    from discovery.likelihood_metamath import ffunc
    closure_clogl = ffunc(vsm.make_kernelproduct_gpcomponent(
        ys, transform=[_legacy_decenter_transform(model)]))

    transport_clogl = model.clogL

    def val(fn, p):
        out = fn(p)
        return float(out[0]) if isinstance(out, tuple) else float(out)

    coeff = [p for p in transport_clogl.params if "_coefficients(" in p]
    hyper = [p for p in transport_clogl.params if p not in coeff]

    np.random.seed(20260716)
    for _ in range(20):
        p0 = ds.sample_uniform(hyper)
        for key in coeff:
            p0[key] = np.random.randn(int(key[key.index("(") + 1:key.index(")")]))
        np.testing.assert_allclose(val(transport_clogl, p0),
                                   val(closure_clogl, p0), rtol=1e-12)


# ==========================================================================
# helpers for the standalone-block tests
# ==========================================================================

def _commongp(psrs, components=10):
    return ds.makecommongp_fourier(psrs, ds.powerlaw, components=components,
                                   T=ds.getspan(psrs), name="rednoise")


def _globalgp(psrs, components=5):
    return ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf,
                                   components=components, T=ds.getspan(psrs),
                                   name="gw")


def _transport_for(psr, psrs, i, *, center=True, extra_globalgp=None):
    gp = _commongp(psrs)
    blocks = [tr.gp_block(gp, psr_slot=i)]
    if extra_globalgp is not None:
        blocks.append(tr.globalgp_curn_block(extra_globalgp, i, len(psrs)))
    return tr.Transport(
        blocks,
        reference_noise=tr.reference_noise(psr),
        reference_residual=np.asarray(psr.residuals) if center else None,
        center=center)


# ==========================================================================
# 2. Jacobian
# ==========================================================================

@pytest.mark.parametrize("center", [False, True])
def test_jacobian_matches_returned_ldJ(psr, metamath_backend, center):
    """jac slogdet of xi -> q equals the returned ldJ (rtol=1e-10), with and
    without centering (centering is a translation and must not change ldJ)."""
    psrs = [psr]
    t = _transport_for(psr, psrs, 0, center=center)
    params = ds.sample_uniform(t.params)

    def q_of_xi(xi):
        return t.apply(params, xi)[0]

    xi = kh.jnparray(np.random.default_rng(0).normal(size=t.dimension))
    J = jax.jacfwd(q_of_xi)(xi)
    sign, logdet = np.linalg.slogdet(np.asarray(J))

    _, ldJ = t.apply(params, xi)

    assert sign > 0
    np.testing.assert_allclose(logdet, float(ldJ), rtol=1e-10)


# ==========================================================================
# 3. Reparam contract — true .params
# ==========================================================================

def test_as_reparam_declares_the_true_params(psrs, metamath_backend):
    """The `decenter_transform.params = []` defect dies: the transport's reparam
    reports the union of its blocks' conditioner hyperparameters."""
    gp = _commongp(psrs)
    t = _transport_for(psrs[0], psrs, 0)

    expected = sorted(set(gp.Phi.getN.params))
    assert t.as_reparam().params == expected
    assert expected           # a powerlaw commongp has hyperparameters


# ==========================================================================
# 4. Scale matching — the anti-ridge regression
# ==========================================================================

def _unit_column_ref(n_toa, n0_scalar):
    """Reference-noise solve for a diagonal N0 = n0_scalar * I over `n_toa`."""
    n0 = np.full(n_toa, float(n0_scalar))

    def solve(rhs):
        rhs = np.asarray(rhs)
        d = n0[:, None] if rhs.ndim == 2 else n0
        return rhs / d, float(np.sum(np.log(n0)))

    return tr._FrozenSolve(solve, f"diag({n0_scalar})")


def _one_by_one_A(transport):
    """The scalar precision A = L^2 of a 1-block, 1-column transport at params={}."""
    cf, _ = transport._factor({})
    L = np.asarray(cf[0])
    return float(L[0, 0]) ** 2


def test_anti_ridge_whitening_at_exact_precision_only(metamath_backend):
    """The anti-ridge regression, tested on the TRANSFORMED POSTERIOR directly
    (not via the diagnostics metric, which structurally cannot see a ridge --
    it builds its target with the block's own precision, so the precision
    cancels in H/A).

    Physical posterior precision M = lambda + p (data curvature + TRUE prior
    precision). The transport whitens with A = lambda0 + conditioner. In xi
    coordinates (q = L^-T xi, A = L L^T) the transformed posterior eigenvalue is
    M / A. With the reference noise equal to the true noise (lambda0 = lambda):

        conditioner = p      -> M/A = (lambda+p)/(lambda+p)     = 1     (whitened)
        conditioner = 100 p  -> M/A = (lambda+p)/(lambda+100 p) != 1

    All quantities are O(1) so the two cases are far apart: a transport that
    floored/ridged the conditioner, or ignored it, changes A and fails here.
    """
    rng = np.random.default_rng(1)
    n_toa = 16
    w = rng.normal(size=(n_toa, 1))
    w = w / np.linalg.norm(w)                     # w^T w = 1  ->  lambda = 1
    ref = _unit_column_ref(n_toa, 1.0)            # N0 = N = I  ->  lambda0 = lambda

    lam = float(w[:, 0] @ w[:, 0])                # = 1 (unit column, N=I)
    p_true = 1.0
    M_phys = lam + p_true                         # TRUE posterior precision (indep. of conditioner)

    def transport(precision):
        return tr.Transport([_const_block(w, [precision], name="s")],
                            reference_noise=ref, center=False)

    # exact conditioner -> whitened.
    A_exact = _one_by_one_A(transport(p_true))
    np.testing.assert_allclose(M_phys / A_exact, 1.0, rtol=1e-10)

    # 100x-wrong conditioner -> (lambda+p)/(lambda+alpha), emphatically NOT 1.
    alpha = 100 * p_true
    A_wrong = _one_by_one_A(transport(alpha))
    np.testing.assert_allclose(M_phys / A_wrong, (lam + p_true) / (lam + alpha),
                               rtol=1e-10)
    assert abs(M_phys / A_wrong - 1.0) > 0.9      # genuinely un-whitened, no swamping


def test_diagnostics_metric_reports_reference_noise_mismatch(metamath_backend):
    """Honest coverage of `diagnostics(noise_solve=...)`. The metric it returns is
    (lambda + p)/(lambda0 + p) with lambda from the live noise solve and lambda0
    from the frozen reference -- i.e. it measures how well the reference noise N0
    approximates the true N (at the block's scale), and is exactly 1.0 when
    N0 = N. It does NOT (and cannot) reveal a conditioner ridge; the anti-ridge
    property is covered by `test_anti_ridge_whitening_at_exact_precision_only`.
    """
    rng = np.random.default_rng(3)
    n_toa = 16
    w = rng.normal(size=(n_toa, 1))
    w = w / np.linalg.norm(w)                     # w^T w = 1
    p = 3.0

    t = tr.Transport([_const_block(w, [p], name="s")],
                     reference_noise=_unit_column_ref(n_toa, 2.0), center=False)  # N0 = 2 I

    # true noise N = I  ->  lambda = 1, lambda0 = 1/2
    def true_solve(rhs):
        rhs = np.asarray(rhs)
        return rhs, 0.0                           # N = I

    d = t.diagnostics(params={}, noise_solve=true_solve)
    lam, lam0 = 1.0, 0.5
    np.testing.assert_allclose(d["metric_eig_min"], (lam + p) / (lam0 + p), rtol=1e-8)
    assert d["metric_kind"] == "local_target"     # no CURN block present

    # N0 == N  ->  metric exactly 1.
    d_match = t.diagnostics(params={},
                            noise_solve=lambda rhs: (np.asarray(rhs) / 2.0, 0.0))
    np.testing.assert_allclose(d_match["metric_eig_min"], 1.0, rtol=1e-8)


# ==========================================================================
# 5. Adapter correctness
# ==========================================================================

def test_gp_block_selects_the_pulsar_row(psrs, metamath_backend):
    gp = _commongp(psrs)
    params = ds.sample_uniform(gp.Phi.getN.params)

    block = tr.gp_block(gp, psr_slot=1)
    got = np.asarray(block.conditioner_precision(params))

    expected = np.asarray(gp.Phi.getN(params))[1] ** -1
    np.testing.assert_allclose(got, expected, rtol=1e-12)

    k = np.asarray(gp.F[1]).shape[1]
    assert got.shape == (k,)
    assert len(block.index) == 1
    (key, sli), = block.index.items()
    assert psrs[1].name in key and sli == slice(0, k)


def test_globalgp_curn_block_matches_the_reshaped_diagonal(psrs, metamath_backend):
    gg = _globalgp(psrs)
    params = ds.sample_uniform(gg.Phi.getN.params)

    block = tr.globalgp_curn_block(gg, 1, len(psrs))
    got = np.asarray(block.conditioner_precision(params))

    expected = np.asarray(
        kh.jnp.diag(gg.Phi.getN(params) ** -1).reshape((len(psrs), -1))[1])
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert block.conditioner_kind == "curn_inverse_marginal"


# ==========================================================================
# 6. Validation errors (each asserts the message content)
# ==========================================================================

def _diag_ref(psr):
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2

    def solve(rhs):
        rhs = np.asarray(rhs)
        d = n0[:, None] if rhs.ndim == 2 else n0
        return rhs / d, float(np.sum(np.log(n0)))

    return tr._FrozenSolve(solve, "diag")


def _const_block(F, precision, name="b", kind="exact_diagonal"):
    F = np.asarray(F, dtype=np.float64)
    prec = np.atleast_1d(np.asarray(precision, dtype=np.float64))

    def cond(params):
        return kh.jnp.asarray(prec)
    cond.params = []
    return tr.TransportBlock(name, F, {f"{name}_coefficients({F.shape[1]})":
                                       slice(0, F.shape[1])}, cond, kind)


def test_zero_column_basis_raises_naming_indices(psr, metamath_backend):
    """The adapters' construction-time column validator (`_validate_columns`)
    names the offending column index. It is the gate the GP adapters run before
    building a block."""
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match=r"zero or non-finite column.*\[1\]"):
        tr._validate_columns("rednoise", F)


def test_empty_block_list_raises(psr, metamath_backend):
    with pytest.raises(ValueError, match="at least one block"):
        tr.Transport([], reference_noise=_diag_ref(psr), center=False)


def test_multi_key_block_index_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]

    def cond(params):
        return kh.jnp.array([1.0])
    cond.params = []
    bad = tr.TransportBlock("b", F, {"a(1)": slice(0, 1), "b(1)": slice(0, 1)},
                            cond)
    with pytest.raises(ValueError, match="exactly one coefficient key"):
        tr.Transport([bad], reference_noise=_diag_ref(psr), center=False)


def test_duplicate_coefficient_key_across_blocks_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    b1 = _const_block(F, [1.0], name="dup")
    b2 = _const_block(F, [1.0], name="dup")
    with pytest.raises(ValueError, match="duplicate coefficient key"):
        tr.Transport([b1, b2], reference_noise=_diag_ref(psr), center=False)


def test_non_localized_slice_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]

    def cond(params):
        return kh.jnp.array([1.0])
    cond.params = []
    bad = tr.TransportBlock("b", F, {"b_coefficients(1)": slice(3, 4)}, cond)
    with pytest.raises(ValueError, match="localized to slice"):
        tr.Transport([bad], reference_noise=_diag_ref(psr), center=False)


def test_ntoa_mismatch_raises(psr, metamath_backend):
    F1 = np.asarray(psr.residuals)[:, None]
    F2 = np.asarray(psr.residuals)[:-1, None]
    with pytest.raises(ValueError, match="disagree on n_toa"):
        tr.Transport([_const_block(F1, [1.0], name="a"),
                      _const_block(F2, [1.0], name="b")],
                     reference_noise=_diag_ref(psr), center=False)


def test_center_true_without_residual_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    with pytest.raises(ValueError, match="center=True requires reference_residual"):
        tr.Transport([_const_block(F, [1.0])], reference_noise=_diag_ref(psr),
                     center=True)


def test_frozen_kernel_with_free_params_lists_missing(psr, metamath_backend):
    kernel = ds.makenoise_measurement(psr, {})       # free EFAC/EQUAD
    with pytest.raises(ValueError, match=r"missing.*efac"):
        tr.reference_noise_frozen(kernel, params0={})


def test_non_metamath_kernel_to_frozen_raises_typeerror(psr):
    ds.config(kernels="matrix")
    try:
        kernel = ds.makenoise_measurement(psr, psr.noisedict)
        with pytest.raises(TypeError, match="metamath kernel"):
            tr.reference_noise_frozen(kernel, params0=psr.noisedict)
    finally:
        ds.config(kernels="matrix")


def test_reference_noise_diagonal_quadratic_and_std(psr, metamath_backend):
    """The diagonal TOA-error reference exposes an exact diag(N0), and the
    transport's N0^-1 quadratic and sqrt(diag(N0)) helpers agree with it
    (geometry certifier support)."""
    ref = tr.reference_noise(psr)
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2
    assert np.allclose(np.asarray(ref.diagonal()), n0)

    F = np.asarray(psr.residuals)[:, None]
    transport = tr.Transport(
        [_const_block(F, [1.0])], reference_noise=ref, center=False)
    n_toa = n0.shape[0]
    v = np.linspace(-1.0, 1.0, n_toa) * 1e-6
    assert np.isclose(float(transport.reference_noise_quadratic(v)),
                      float(v @ (v / n0)))
    assert np.allclose(
        np.asarray(transport.reference_noise_standard_deviation()), np.sqrt(n0))


def test_frozen_kernel_diagonal_adds_ecorr_exposure(metamath_backend):
    """diag(N0) for a Sherman-Morrison ECORR reference is diag(N) + F P F^T's
    diagonal (F is 0/1 exposure, so it adds F P epoch-wise)."""
    from discovery import metamath as mm

    N = np.array([1.0, 2.0, 3.0, 4.0])
    F = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    P = np.array([10.0, 20.0])
    kernel = mm.NoiseMatrixSM(kh.jnparray(N), F, kh.jnparray(P))
    diag = tr._frozen_kernel_diagonal(kernel, {})
    assert np.allclose(diag, N + F @ P)


def test_frozen_measurement_reference_diagonal_is_finite_positive(
        psr, metamath_backend):
    """A frozen measurement-noise reference (real EFAC/EQUAD/ECORR at the
    noisedict) yields a strictly positive finite diag(N0) of TOA length, and
    the transport's quadratic/std helpers stay finite."""
    kernel = ds.makenoise_measurement(psr, psr.noisedict)
    ref = tr.reference_noise_frozen(kernel, params0=psr.noisedict)
    diag = np.asarray(ref.diagonal())
    n_toa = np.asarray(psr.toaerrs).shape[0]
    assert diag.shape == (n_toa,)
    assert np.all(np.isfinite(diag)) and np.all(diag > 0.0)

    F = np.asarray(psr.residuals)[:, None]
    transport = tr.Transport(
        [_const_block(F, [1.0])], reference_noise=ref, center=False)
    std = np.asarray(transport.reference_noise_standard_deviation())
    assert np.allclose(std, np.sqrt(diag))
    v = np.linspace(-1.0, 1.0, n_toa) * 1e-6
    assert np.isfinite(float(transport.reference_noise_quadratic(v)))


def test_fourier_sensitivity_weights_with_frozen_reference_noise(
        psr, metamath_backend):
    """The pivot sensitivity weights run against a real frozen
    measurement-noise reference and a real Fourier basis, not a toy solver."""
    from discovery import signals as sig

    f, _df, fmat = sig.fourierbasis(psr, 5)
    ref = tr.reference_noise_frozen(
        ds.makenoise_measurement(psr, psr.noisedict), psr.noisedict)
    w = sig.fourier_sensitivity_weights(fmat, ref)
    assert w.shape == (5,)
    assert np.all(np.isfinite(w)) and np.all(w > 0.0)
    # The sensitivity-weighted pivot lands inside the sampled frequency band.
    freqs = np.asarray(f)[0::2]
    f_pivot = sig.sensitivity_weighted_pivot_frequency(freqs, w)
    assert freqs.min() <= f_pivot <= freqs.max()


def test_matrix_mode_transport_construction_raises(psr):
    ds.config(kernels="matrix")
    try:
        F = np.asarray(psr.residuals)[:, None]
        with pytest.raises(NotImplementedError, match="metamath kernel path"):
            tr.Transport([_const_block(F, [1.0])], reference_noise=_diag_ref(psr),
                         center=False)
    finally:
        ds.config(kernels="matrix")


def test_validate_negative_precision_raises_with_local_indices(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 1.0]])
    t = tr.Transport([_const_block(F, [1.0, -2.0], name="neg")],
                     reference_noise=_diag_ref(psr), center=False)
    with pytest.raises(ValueError, match=r"negative.*\[1\]"):
        t.validate({})


def test_validate_rank_deficient_basis_with_zero_precision_fails_pd(psr, metamath_backend):
    """Two NONZERO collinear columns (F[:,1] = 2 F[:,0]) make G0 rank-1. With
    zero conditioner precision the null direction is unconstrained and A is
    singular, so the PD check fails. A strictly positive diagonal precision
    would add eps*I and correctly keep A PD -- the point of forbidding a floor:
    the pathology must surface, not be silently patched."""
    c0 = np.asarray(psr.residuals)[:, None]
    F = np.concatenate([c0, 2.0 * c0], axis=1)
    t = tr.Transport([_const_block(F, [0.0, 0.0], name="rank")],
                     reference_noise=_diag_ref(psr), center=False)
    with pytest.raises(ValueError, match="not positive definite"):
        t.validate({})

    # a strictly positive diagonal precision restores PD.
    t_pd = tr.Transport([_const_block(F, [1e-3, 1e-3], name="rank")],
                        reference_noise=_diag_ref(psr), center=False)
    t_pd.validate({})   # does not raise


def test_globalgp_curn_block_mismatched_npsr_raises(psrs, metamath_backend):
    gg = _globalgp(psrs)
    with pytest.raises(ValueError, match="npsr="):
        tr.globalgp_curn_block(gg, 0, len(psrs) + 1)


def test_globalgp_curn_block_out_of_range_slot_raises(psrs, metamath_backend):
    gg = _globalgp(psrs)
    with pytest.raises(ValueError, match="out of range"):
        tr.globalgp_curn_block(gg, len(psrs), len(psrs))


def test_array_transport_unequal_dimensions_raises(psrs, metamath_backend):
    a = _transport_for(psrs[0], psrs, 0)
    # a second transport with a DIFFERENT dimension (fewer components)
    gp_small = ds.makecommongp_fourier(psrs, ds.powerlaw, components=5,
                                       T=ds.getspan(psrs), name="rednoise")
    b = tr.Transport([tr.gp_block(gp_small, psr_slot=1)],
                     reference_noise=tr.reference_noise(psrs[1]),
                     reference_residual=np.asarray(psrs[1].residuals))
    with pytest.raises(ValueError, match="equal per-pulsar dimension"):
        tr.ArrayTransport([a, b])


def test_array_transport_mixed_centering_raises(psrs, metamath_backend):
    a = _transport_for(psrs[0], psrs, 0, center=True)
    b = _transport_for(psrs[1], psrs, 1, center=False)
    with pytest.raises(ValueError, match="all-or-none centering"):
        tr.ArrayTransport([a, b])


def test_reparam_shape_mismatch_raises(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)])
    rp = at.as_reparam()
    wrong = kh.jnparray(np.zeros((at.npsr, at.dimension + 1)))
    with pytest.raises(ValueError, match="coefficient array has shape"):
        rp({}, wrong)


def test_arraylikelihood_decenter_and_transport_mutually_exclusive(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)])
    with pytest.raises(ValueError, match="mutually exclusive"):
        ds.ArrayLikelihood([ds.PulsarLikelihood(
            [p.residuals, ds.makenoise_measurement(p, p.noisedict)]) for p in psrs],
            commongp=_commongp(psrs), decenter=True, transport=at)


def test_arraylikelihood_wrong_transport_pulsar_count_raises(psrs, metamath_backend):
    at = tr.ArrayTransport([_transport_for(p, psrs, i)
                            for i, p in enumerate(psrs)][:2])
    with pytest.raises(ValueError, match="pulsars"):
        ds.ArrayLikelihood([ds.PulsarLikelihood(
            [p.residuals, ds.makenoise_measurement(p, p.noisedict)]) for p in psrs],
            commongp=_commongp(psrs), transport=at)


# ==========================================================================
# 7. Reference-noise geometry, without a false density invariant
# ==========================================================================

def test_inverse_map_round_trip_and_shared_target(psr, metamath_backend):
    """Build two 2-D transports (toaerr-diagonal vs frozen noisedict). For each,
    invert q = mu + L^-T xi via xi = L^T (q - mu) and recover the same physical
    q. The target logp(q) is identical because it is the same likelihood. Do NOT
    assert equality of logp(q)+ldJ across transports: those are densities w.r.t.
    different xi coordinates and their Jacobians legitimately differ."""
    psrs = [psr]
    gp = _commongp(psrs)
    params = ds.sample_uniform(gp.Phi.getN.params)

    def build(ref):
        return tr.Transport([tr.gp_block(gp, psr_slot=0)], reference_noise=ref,
                            center=False)

    t_diag = build(tr.reference_noise(psr))
    t_frozen = build(tr.reference_noise_frozen(
        ds.makenoise_measurement(psr, psr.noisedict), params0=psr.noisedict))

    q_phys = kh.jnparray(np.random.default_rng(3).normal(size=t_diag.dimension))

    for t in (t_diag, t_frozen):
        cf, _ = t._factor(params)
        # xi = L^T (q - mu); here center=False so mu=0
        xi = np.asarray(cf[0]).T @ np.asarray(q_phys)
        q_back, _ = t.apply(params, kh.jnparray(xi))
        np.testing.assert_allclose(np.asarray(q_back), np.asarray(q_phys), rtol=1e-8)


# ==========================================================================
# 8. Parameter-dependent basis rejection
# ==========================================================================

def test_parameter_dependent_basis_is_rejected(psrs, metamath_backend):
    """A graph-valued basis carrying a live parameter is refused, naming it."""
    from collections import OrderedDict

    def getF(params):
        return np.ones((5, 2)) * params["x_live"]
    getF.params = ["x_live"]
    live_graph = OrderedDict([("F", metamatrix.FuncLeaf(fn=getF))])

    with pytest.raises(ValueError, match=r"depends on parameters \['x_live'\]"):
        tr._eval_basis(live_graph)


# ==========================================================================
# free-EFAC under decenter=True
# ==========================================================================

def _psl_freewn(psr):
    return ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, {})])


def test_decenter_with_free_efac_raises_naming_missing_params(psrs, metamath_backend):
    """decenter=True freezes each per-pulsar kernel at params0={}; a free EFAC
    makes that freeze incomplete, converting the old closure's silent
    constant-N assumption into a diagnosed error."""
    model = ds.ArrayLikelihood([_psl_freewn(p) for p in psrs],
                               commongp=_commongp(psrs), decenter=True)
    with pytest.raises(ValueError, match=r"missing.*efac"):
        model.clogL


def test_free_efac_via_explicit_transport_evaluates_finitely(psrs, metamath_backend):
    """The same varying-white-noise model builds and evaluates when the
    transport uses a dependency-free reference_noise(psr)."""
    per_psr = []
    for i, p in enumerate(psrs):
        per_psr.append(tr.Transport(
            [tr.gp_block(_commongp(psrs), psr_slot=i)],
            reference_noise=tr.reference_noise(p),
            reference_residual=np.asarray(p.residuals), center=True))
    at = tr.ArrayTransport(per_psr)

    model = ds.ArrayLikelihood([_psl_freewn(p) for p in psrs],
                               commongp=_commongp(psrs), transport=at)

    clogl = model.clogL
    coeff = [p for p in clogl.params if "_coefficients(" in p]
    p0 = ds.sample_uniform([p for p in clogl.params if p not in coeff])
    rng = np.random.default_rng(0)
    for key in coeff:
        p0[key] = rng.normal(size=int(key[key.index("(") + 1:key.index(")")]))

    out = clogl(p0)
    logp = float(out[0]) if isinstance(out, tuple) else float(out)
    assert np.isfinite(logp)


# ==========================================================================
# pathology matrix — legal extreme still factorizes
# ==========================================================================

def test_legal_extreme_still_factorizes(psrs, metamath_backend):
    """validate passes at a nominal draw; then a draw that drives the stochastic
    block's precision to a legal extreme (log10_A at its prior bound) still
    factorizes -- PD holds because the exact prior precision is present."""
    t = _transport_for(psrs[0], psrs, 0, center=False)

    nominal = ds.sample_uniform(t.params)
    t.validate(nominal)                              # does not raise

    extreme = dict(nominal)
    for key in t.params:
        if key.endswith("_log10_A"):
            lo, hi = ds.getprior_uniform(key)
            extreme[key] = hi                        # smallest prior variance
    diag = t.validate(extreme)                       # still PD
    assert diag["chol_diag_min"] > 0.0


# ==========================================================================
# serialization position
# ==========================================================================

def test_transport_is_not_picklable(psrs, metamath_backend):
    """Transports hold JAX arrays and closures; no pickle support is promised
    Local closures typically raise AttributeError under stdlib pickle."""
    import pickle
    t = _transport_for(psrs[0], psrs, 0)
    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        pickle.dumps(t)


@pytest.mark.slow
def test_checkpoint_run_with_a_transport_reparam(psrs, metamath_backend, tmp_path):
    """run_nuts_with_checkpoints completes a two-checkpoint run for a model whose
    clogL contains a transport reparam: the sampler-state pickle never touches
    model objects."""
    import jax
    import pandas as pd
    import numpyro
    from numpyro import distributions as dist
    from discovery import prior as ds_prior
    from discovery.samplers import numpyro as ds_numpyro

    model = R.decenter_intrinsic_rn(psrs)
    clogl = model.clogL                              # carries the transport reparam
    assert model.transport is None and model.decenter

    coeff = [p for p in clogl.params if "_coefficients(" in p]
    hyper = [p for p in clogl.params if p not in coeff]

    def numpyro_model():
        params = {p: numpyro.sample(p, dist.Uniform(*ds_prior.getprior_uniform(p)))
                  for p in hyper}
        # decentered coefficients are whitened -> standard-normal xi
        for key in coeff:
            width = int(key[key.index("(") + 1:key.index(")")])
            params[key] = numpyro.sample(key, dist.Normal(0.0, 1.0).expand([width]))
        out = clogl(params)
        numpyro.factor("logl", out[0] if isinstance(out, tuple) else out)
    # feather can't hold the vector coefficient columns; the point is
    # that the checkpoint run completes and pickles sampler state (not model
    # objects), so a scalar-only frame is sufficient here.
    import numpy as _np
    numpyro_model.to_df = lambda chain: pd.DataFrame(
        {k: _np.asarray(v) for k, v in chain.items() if _np.asarray(v).ndim == 1})

    sampler = ds_numpyro.makesampler_nuts(numpyro_model, num_warmup=4,
                                          num_samples=4, progress_bar=False)
    outdir = tmp_path / "chains"
    ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                         outdir=outdir)

    assert len(pd.read_feather(outdir / "numpyro-samples.feather")) == 4


# ==========================================================================
# Transport.fingerprint — structural digest for run-manifest reconciliation
# (Track J: consumed by nltiming's dynamic run writer)
# ==========================================================================


def test_transport_fingerprint_is_stable_and_structural(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5]])
    ref = _diag_ref(psr)

    a = tr.Transport([_const_block(F, [1.0, 1.0], name="rn")],
                     reference_noise=ref, center=False)
    # Same structure -> identical fingerprint (independent of a params draw).
    b = tr.Transport([_const_block(F, [2.0, 3.0], name="rn")],
                     reference_noise=ref, center=False)
    assert a.fingerprint().startswith("sha256:")
    assert a.fingerprint() == b.fingerprint()

    # A different block name is a structural change -> different fingerprint.
    c = tr.Transport([_const_block(F, [1.0, 1.0], name="dm")],
                     reference_noise=ref, center=False)
    assert c.fingerprint() != a.fingerprint()

    # Different dimensionality -> different fingerprint.
    F1 = np.asarray(psr.residuals)[:, None] * np.array([[1.0]])
    d = tr.Transport([_const_block(F1, [1.0], name="rn")],
                     reference_noise=ref, center=False)
    assert d.fingerprint() != a.fingerprint()


def test_array_transport_fingerprint_is_stable(psrs, metamath_backend):
    at1 = tr.ArrayTransport([_transport_for(p, psrs, i)
                             for i, p in enumerate(psrs)])
    at2 = tr.ArrayTransport([_transport_for(p, psrs, i)
                             for i, p in enumerate(psrs)])
    assert at1.fingerprint().startswith("sha256:")
    assert at1.fingerprint() == at2.fingerprint()


# ==========================================================================
# extensions: array_block, ExtSignal centering, softclip
# ==========================================================================


class _FakeExtSignal:
    """Minimal ExtSignal duck for center_extsignals tests: .Fs, .coeffs, .name."""

    def __init__(self, Fs, kext, name="cw"):
        self.Fs, self.name = Fs, name

        def coeffs(params):
            return kh.jnp.asarray(
                [[params[f"{name}_c{j}"] for j in range(kext)] for _ in Fs])

        coeffs.params = [f"{name}_c{j}" for j in range(kext)]
        self.coeffs = coeffs


def test_array_block_precision_specs(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5, -0.3]])
    # scalar broadcast
    b = tr.array_block(F, {"tim": slice(0, 3)}, 2.0, name="timing")
    assert b.name == "timing"
    assert list(b.index) == ["tim"]
    assert np.allclose(np.asarray(b.conditioner_precision({})), 2.0)
    assert b.conditioner_precision.params == []
    # (k,) vector
    b2 = tr.array_block(F, {"tim": slice(0, 3)}, [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(b2.conditioner_precision({})), [1.0, 2.0, 3.0])
    # callable with .params

    def cp(params):
        return kh.jnp.asarray([params["s"]] * 3)

    cp.params = ["s"]
    b3 = tr.array_block(F, {"tim": slice(0, 3)}, cp)
    assert b3.conditioner_precision.params == ["s"]
    assert np.allclose(np.asarray(b3.conditioner_precision({"s": 4.0})), 4.0)


def test_array_block_validation(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.5]])
    with pytest.raises(ValueError, match="one-key dict"):
        tr.array_block(F, {"a": slice(0, 2), "b": slice(0, 2)}, 1.0)
    with pytest.raises(ValueError, match=r"slice\(0, 2\)"):
        tr.array_block(F, {"a": slice(0, 1)}, 1.0)
    with pytest.raises(ValueError, match="non-finite or negative"):
        tr.array_block(F, {"a": slice(0, 2)}, [-1.0, 1.0])
    Fbad = np.asarray(psr.residuals)[:, None] * np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero or non-finite"):
        tr.array_block(Fbad, {"a": slice(0, 2)}, 1.0)


def test_extsignal_subtracted_centering_matches_manual(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    ntoa = r0.shape[0]
    F = r0[:, None] * np.array([[1.0, 0.5, -0.3]])
    rng = np.random.default_rng(3)
    Fext = rng.standard_normal((ntoa, 2))
    es = _FakeExtSignal([Fext], 2, name="cw")
    ref = _diag_ref(psr)

    t = tr.Transport([tr.array_block(F, {"tim": slice(0, 3)}, 1.0, name="timing")],
                     reference_noise=ref, reference_residual=r0, center=True,
                     center_extsignals=[es], psr_slot=0)
    assert set(["cw_c0", "cw_c1"]).issubset(set(t.params))

    params = {"cw_c0": 0.7, "cw_c1": -0.3}
    q, _ = t.apply(params, kh.jnp.zeros(3))

    n0 = np.asarray(psr.toaerrs) ** 2
    A = F.T @ (F / n0[:, None]) + np.eye(3)
    b0 = F.T @ (r0 / n0)
    E0 = F.T @ (Fext / n0[:, None])
    mu = np.linalg.solve(A, b0 - E0 @ np.array([0.7, -0.3]))
    assert np.allclose(np.asarray(q), mu)
    assert t.diagnostics()["center_extsignals"] == ["cw"]


def test_softclip_clamps_centering_slice(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    F = r0[:, None]                                   # 1-column, near-collinear
    ref = _diag_ref(psr)
    # A pathological reference residual drives mu far outside [-4, 4].
    big = np.zeros_like(r0)
    big[0] = 1e6
    t = tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1e-10, name="timing")],
                     reference_noise=ref, reference_residual=big, center=True,
                     softclip={"timing": 4.0})
    q, _ = t.apply({}, kh.jnp.zeros(1))
    assert abs(float(np.asarray(q)[0])) <= 4.0 + 1e-9
    assert t.diagnostics()["softclip"] == {"timing": 4.0}


def test_softclip_and_extsignals_require_center(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    ref = _diag_ref(psr)
    with pytest.raises(ValueError, match="softclip requires center"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                     reference_noise=ref, center=False, softclip={"timing": 4.0})
    es = _FakeExtSignal([np.asarray(psr.residuals)[:, None]], 1)
    with pytest.raises(ValueError, match="center_extsignals requires center"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0)],
                     reference_noise=ref, center=False, center_extsignals=[es])


def test_softclip_unknown_block_raises(psr, metamath_backend):
    F = np.asarray(psr.residuals)[:, None]
    ref = _diag_ref(psr)
    with pytest.raises(ValueError, match="unknown block 'nope'"):
        tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                     reference_noise=ref, reference_residual=np.asarray(psr.residuals),
                     center=True, softclip={"nope": 4.0})


def test_array_transport_rejects_extsignals_and_softclip(psr, metamath_backend):
    r0 = np.asarray(psr.residuals, dtype=float)
    F = r0[:, None]
    ref = _diag_ref(psr)
    t = tr.Transport([tr.array_block(F, {"tim": slice(0, 1)}, 1.0, name="timing")],
                     reference_noise=ref, reference_residual=r0, center=True,
                     softclip={"timing": 4.0})
    with pytest.raises(ValueError, match="does not support"):
        tr.ArrayTransport([t])


# ==========================================================================
# MarginalTransport
# ==========================================================================

from discovery import metamath  # noqa: E402


class _MockPsr:
    """Minimal pulsar for the GP signal factories (name / pos / toas)."""

    name = "J0000+0000"

    def __init__(self, n, span_days=3000.0):
        self.toas = 53000.0 * 86400.0 + np.linspace(0.0, span_days * 86400.0, n)
        self.pos = np.array([1.0, 0.0, 0.0])


def _build_toy_marginal(rng, n=40, comps=3):
    """Fold WN + powerlaw RN + improper timing (1e40) + unit-normal z-prior into
    one metamath WoodburyKernel stack (exactly the shape of a marginalized
    ``PulsarLikelihood.N``), plus the pieces for an independent NumPy oracle."""
    psr = _MockPsr(n)
    n0 = rng.uniform(0.5, 1.5, n)
    gp_rn = ds.makegp_fourier(psr, ds.powerlaw, comps, name="rednoise")
    gp_imp = ds.makegp_improper(psr, rng.standard_normal((n, 2)), name="tm")
    gp_sn = ds.makegp_standard_normal(psr, rng.standard_normal((n, 2)), name="zprior")

    K = metamath.NoiseMatrix(kh.jnparray(n0))
    Fs, Phis = [], []
    for gp in (gp_rn, gp_imp, gp_sn):
        F = np.asarray(gp.F, dtype=np.float64)
        K = metamath.WoodburyKernel(K, kh.jnparray(F), gp.Phi)
        Fs.append(F)
        Phis.append(gp.Phi)
    F_all = np.concatenate(Fs, axis=1)

    def phi_of(eta):
        out = []
        for Ph in Phis:
            c, f = metamath._materialize(Ph.N)
            out.append(np.asarray(c if f is None else f(params=eta), dtype=np.float64))
        return np.concatenate(out)

    eta_names = list(gp_rn.Phi.N.params)  # powerlaw log10_A / gamma
    return dict(n0=n0, K=K, F_all=F_all, phi_of=phi_of, eta_names=eta_names)


def _eta(names, rng):
    out = {}
    for nm in names:
        out[nm] = float(rng.uniform(-1.0, 0.0)) if "log10_A" in nm else float(rng.uniform(1.5, 5.0))
    return out


def _oracle_products(n0, F_all, phi, W, y):
    """Independent NumPy Woodbury: (W^T C^-1 W, W^T C^-1 y) for
    C = diag(n0) + F diag(phi) F^T. Stable with a 1e40 improper entry
    (Phi^-1 -> 1e-40), unlike a dense inverse of C."""
    Ninv = 1.0 / n0
    NiF = Ninv[:, None] * F_all
    NiW = Ninv[:, None] * W
    Niy = Ninv * y
    inner = np.diag(1.0 / phi) + F_all.T @ NiF
    cfa = np.linalg.cholesky(inner)

    def isolve(rhs):
        return np.linalg.solve(cfa.T, np.linalg.solve(cfa, rhs))

    WtNiF = W.T @ NiF
    WtCiW = W.T @ NiW - WtNiF @ isolve(F_all.T @ NiW)
    WtCiy = W.T @ Niy - WtNiF @ isolve(F_all.T @ Niy)
    return WtCiW, WtCiy


def _oracle_quadratic(n0, F_all, phi, v):
    Ninv = 1.0 / n0
    Niv = Ninv * v
    NiF = Ninv[:, None] * F_all
    inner = np.diag(1.0 / phi) + F_all.T @ NiF
    cfa = np.linalg.cholesky(inner)
    FtNiv = F_all.T @ Niv
    return float(v @ Niv - FtNiv @ np.linalg.solve(cfa.T, np.linalg.solve(cfa, FtNiv)))


def _make_block(W):
    key = f"{_MockPsr.name}_timing_timing_z"
    return tr.array_block(W, {key: slice(0, W.shape[1])},
                          conditioner_precision=1.0, name="timing"), key


def test_marginal_transport_matches_dense_oracle(metamath_backend):
    """T-D1: A(eta) and mu(eta) from MarginalTransport._factor match the
    independent Woodbury oracle W^T C^-1 W + I / A^-1 W^T C^-1 y at 5 eta draws."""
    rng = np.random.default_rng(0)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], y, block, center=True)

    for _ in range(5):
        eta = _eta(toy["eta_names"], rng)
        phi = toy["phi_of"](eta)
        WtCiW, WtCiy = _oracle_products(toy["n0"], toy["F_all"], phi, W, y)
        A_o = WtCiW + np.eye(3)
        mu_o = np.linalg.solve(A_o, WtCiy)

        cf, b, pinv = t._factor(eta)
        A_t = np.asarray(cf[0]) @ np.asarray(cf[0]).T  # L L^T (lower factor)
        mu_t = np.asarray(t.apply(eta, np.zeros(3))[0])
        assert np.allclose(A_t, A_o, rtol=1e-8, atol=1e-10)
        assert np.allclose(np.asarray(b), WtCiy, rtol=1e-8, atol=1e-10)
        assert np.allclose(mu_t, mu_o, rtol=1e-8, atol=1e-10)


def test_marginal_transport_jacobian_matches_ldj(metamath_backend):
    """T-D2: jacfwd slogdet of z(xi) equals ldJ, center on and off."""
    import jax.numpy as jnp
    rng = np.random.default_rng(1)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    block, _ = _make_block(W)
    eta = _eta(toy["eta_names"], rng)
    for center in (True, False):
        t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block, center=center)
        jac = np.asarray(jax.jacfwd(lambda xi: t.apply(eta, xi)[0])(jnp.zeros(3)))
        _, sld = np.linalg.slogdet(jac)
        ldJ = float(t.apply(eta, jnp.zeros(3))[1])
        assert np.isclose(sld, ldJ, rtol=1e-10, atol=1e-10)


def test_marginal_transport_params_propagation(metamath_backend):
    """T-D3: t.params == union of kernel hyper names and block conditioner params."""
    rng = np.random.default_rng(2)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    block, _ = _make_block(rng.standard_normal((n, 3)))
    t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block)
    assert t.params == sorted(set(toy["eta_names"]) | set(block.conditioner_precision.params))
    assert set(toy["eta_names"]).issubset(set(t.params))


def test_marginal_transport_centering_is_gls_solution(metamath_backend):
    """T-D4: mu(eta) equals the dense GLS solution A^-1 W^T C^-1 y (pins b sign
    and the y_t waveform convention)."""
    rng = np.random.default_rng(3)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], y, block, center=True)
    eta = _eta(toy["eta_names"], rng)
    WtCiW, WtCiy = _oracle_products(toy["n0"], toy["F_all"], toy["phi_of"](eta), W, y)
    gls = np.linalg.solve(WtCiW + np.eye(3), WtCiy)
    mu = np.asarray(t.apply(eta, np.zeros(3))[0])
    assert np.allclose(mu, gls, rtol=1e-8, atol=1e-10)


def test_marginal_transport_failure_semantics(metamath_backend):
    """T-D5: callable y, kernel without make_kernelsolve, multi-key block,
    matrix mode, and negative conditioner precision each fail as specified."""
    rng = np.random.default_rng(4)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    block, _ = _make_block(rng.standard_normal((n, 3)))
    y = rng.standard_normal(n)

    with pytest.raises(TypeError, match="CompoundDelay|D-INV"):
        tr.marginal_transport(toy["K"], (lambda p: y), block)
    with pytest.raises(TypeError, match="make_kernelsolve"):
        tr.marginal_transport(object(), y, block)
    # multi-key index rejected.
    W = rng.standard_normal((n, 3))
    bad = tr.TransportBlock("timing", W,
                            {"a": slice(0, 2), "b": slice(2, 3)},
                            block.conditioner_precision)
    with pytest.raises(ValueError, match="exactly one coefficient key"):
        tr.marginal_transport(toy["K"], y, bad)
    # negative conditioner precision -> validate error (no floor). A constant
    # negative spec is rejected by array_block eagerly, so use a live callable.
    def negcp(params):
        return kh.jnp.asarray([1.0, -1.0, 1.0])
    negcp.params = []
    negblock = tr.TransportBlock("timing", W, {"k": slice(0, 3)}, negcp)
    t = tr.marginal_transport(toy["K"], y, negblock)
    with pytest.raises(ValueError, match="negative|no floor"):
        t.validate(_eta(toy["eta_names"], rng))


def test_marginal_transport_matrix_mode_raises():
    """T-D5 (matrix mode): require_metamath -> NotImplementedError."""
    ds.config(kernels="matrix")
    try:
        rng = np.random.default_rng(5)
        W = rng.standard_normal((10, 2))
        block, _ = _make_block(W)
        with pytest.raises(NotImplementedError, match="metamath"):
            tr.marginal_transport(object(), rng.standard_normal(10), block)
    finally:
        ds.config(kernels="matrix")


def test_marginal_transport_live_kernel_hooks(metamath_backend):
    """T-D6: live_kernel_quadratic and live_kernel_standard_deviation match the
    dense v^T C^-1 v and sqrt(diag C) at two eta points; bad inputs raise."""
    rng = np.random.default_rng(6)
    toy = _build_toy_marginal(rng)
    n = toy["n0"].shape[0]
    W = rng.standard_normal((n, 3))
    block, _ = _make_block(W)
    t = tr.marginal_transport(toy["K"], rng.standard_normal(n), block)

    for _ in range(2):
        eta = _eta(toy["eta_names"], rng)
        phi = toy["phi_of"](eta)
        v = rng.standard_normal(n)
        assert np.isclose(t.live_kernel_quadratic(eta, v),
                          _oracle_quadratic(toy["n0"], toy["F_all"], phi, v),
                          rtol=1e-8, atol=1e-10)
        diag_C = toy["n0"] + (toy["F_all"] ** 2) @ phi
        assert np.allclose(t.live_kernel_standard_deviation(eta),
                           np.sqrt(diag_C), rtol=1e-8, atol=0.0)

    eta = _eta(toy["eta_names"], rng)
    with pytest.raises(ValueError, match="shape"):
        t.live_kernel_quadratic(eta, np.ones(n + 1))


def test_live_kernel_diagonal_type_dispatch(metamath_backend):
    """Type dispatch: WoodburyProjKernel -> NotImplementedError,
    an unsupported kernel type -> TypeError."""
    from discovery.transport import _live_kernel_diagonal

    rng = np.random.default_rng(7)
    n = 20
    psr = _MockPsr(n)
    gp_rn = ds.makegp_fourier(psr, ds.powerlaw, 3, name="rednoise")
    eta = {p: (-0.5 if "log10_A" in p else 3.0) for p in gp_rn.Phi.N.params}

    proj = metamath.WoodburyProjKernel(
        metamath.NoiseMatrix(kh.jnparray(rng.uniform(0.5, 1.5, n))),
        rng.standard_normal((n, 2)),   # M (timing, projected out)
        np.asarray(gp_rn.F),           # F (GP basis, kept)
        gp_rn.Phi,                     # P
    )
    with pytest.raises(NotImplementedError, match="WoodburyProjKernel|projection"):
        _live_kernel_diagonal(proj, eta)
    with pytest.raises(TypeError, match="unsupported kernel type"):
        _live_kernel_diagonal(object(), eta)
