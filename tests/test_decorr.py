"""makegp_decorr: epochs, nesting on makegp_ecorr, the correlated prior and its inverse.

Single-backend pulsars keep their first epoch, and with enterprise=True a backend without
multi-TOA epochs gets an empty basis rather than an error.
"""
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals, likelihood, matrix

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = ["single_backend_pulsar", "multi_backend_pulsar", "empty_epoch_pulsar"]


def _read(name):
    return ds.Pulsar.read_feather(DATA / f"{name}.feather")


def _partition(F):
    """TOAs grouped by the first column each is nonzero in, its epoch's mode-0 column."""
    F = np.asarray(F) != 0
    rows = np.nonzero(F.any(axis=1))[0]
    first = F[rows].argmax(axis=1)
    return sorted(tuple(rows[first == c]) for c in np.unique(first))


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("enterprise", [False, True])
@pytest.mark.parametrize("variant", ["legendre1", "legendre3", "correlated2"])
def test_legendre_epochs_are_the_ecorr_epochs(name, enterprise, variant):
    psr = _read(name)
    ref = signals.makegp_ecorr(psr, enterprise=enterprise).F
    gp = {"legendre1": lambda: signals.makegp_decorr(psr, nmodes=1, enterprise=enterprise),
          "legendre3": lambda: signals.makegp_decorr(psr, nmodes=3, enterprise=enterprise),
          "correlated2": lambda: signals.makegp_decorr(psr, nmodes=2, correlated=True, enterprise=enterprise),
          }[variant]()
    assert _partition(gp.F) == _partition(ref)
    if not enterprise:
        covered = np.asarray(ref) != 0
        assert np.all(covered.sum(axis=1) == (np.asarray(psr.backend_flags) != '').astype(int))


# the single-amplitude ECORR that makegp_ecorr must reproduce exactly
def _upstream_makegp_ecorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=signals.selection_backend_flags, variable=False, name='ecorrGP'):
    log10_ecorrs, Umats = [], []

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')


        if len(np.unique(masks)) == 1:
            first_valid_bin = 0
        else:
            first_valid_bin = 1

        bins = signals.quantize(psr.toas * mask)

        if enterprise:
            uniques, counts = np.unique(bins, return_counts=True)
            epoch_masks = [bins == i for i, cnt in zip(
                uniques[first_valid_bin:],
                counts[first_valid_bin:]) if cnt > 1]

            if epoch_masks:
                U_backend = np.vstack(epoch_masks).T
            else:
                U_backend = np.zeros((len(bins), 0))

            Umats.append(U_backend)
        else:
            Umats.append(np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T)
    Umatall = np.hstack(Umats)
    params = log10_ecorrs

    pmasks, cnt = [], 0
    for Umat in Umats:
        z = np.zeros(Umatall.shape[1], dtype=np.float64)
        z[cnt:cnt+Umat.shape[1]] = 1.0
        pmasks.append(z)
        cnt = cnt + Umat.shape[1]
    logscale = np.log10(scale)

    if all(par in noisedict for par in params):
        phi = sum(10.0**(2 * (logscale + noisedict[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))

        if variable:
            def getphi(params):
                return phi
            getphi.params = []

            gp = signals.matrix.VariableGP(signals.matrix.NoiseMatrix1D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])}
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            return signals.matrix.ConstantGP(signals.matrix.NoiseMatrix1D_novar(phi), Umatall)
    else:
        pmasks = [signals.matrix.jnparray(pmask) for pmask in pmasks]
        def getphi(params):
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))
        getphi.params = params

        gp = signals.matrix.VariableGP(signals.matrix.NoiseMatrix1D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])}
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp


def _white(psr):
    backends = sorted(set(np.asarray(psr.backend_flags).tolist()))
    nd = {f'{psr.name}_{b}_efac': 1.1 for b in backends}
    nd.update({f'{psr.name}_{b}_log10_t2equad': -7.0 for b in backends})
    return signals.makenoise_measurement(psr, noisedict=nd)


def _model(psr, gp):
    return likelihood.PulsarLikelihood([psr.residuals, signals.makegp_timing(psr, svd=True),
                                        _white(psr), gp])


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("enterprise", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("variable", [False, True])
def test_makegp_ecorr_is_bit_identical_to_upstream(name, enterprise, fixed, variable):
    psr = _read(name)
    backends = sorted(b for b in set(np.asarray(psr.backend_flags).tolist()) if b != '')
    nd = {f'{psr.name}_{b}_log10_ecorr': -6.0 + 0.3 * i for i, b in enumerate(backends)} if fixed else {}
    new = signals.makegp_ecorr(psr, noisedict=nd, enterprise=enterprise, variable=variable)
    ref = _upstream_makegp_ecorr(psr, noisedict=nd, enterprise=enterprise, variable=variable)

    assert type(new) is type(ref) and type(new.Phi) is type(ref.Phi)
    assert np.asarray(new.F).dtype == np.asarray(ref.F).dtype
    assert np.array_equal(np.asarray(new.F), np.asarray(ref.F))
    for attr in ('index', 'gpname', 'name', 'gpcommon'):
        assert getattr(new, attr, None) == getattr(ref, attr, None)
    assert list(getattr(new.Phi, 'params', [])) == list(getattr(ref.Phi, 'params', []))
    if hasattr(ref.Phi, 'getN'):
        y = {p: -7.0 + 0.2 * i for i, p in enumerate(ref.Phi.params)}
        assert np.array_equal(np.asarray(new.Phi.getN(y)), np.asarray(ref.Phi.getN(y)))
    else:
        assert np.array_equal(np.asarray(new.Phi.N), np.asarray(ref.Phi.N))

    mn, mr = _model(psr, new), _model(psr, ref)
    assert mn.logL.params == mr.logL.params
    for v in (-7.5, -6.5, -5.5):
        x = {p: v + 0.2 * i for i, p in enumerate(mr.logL.params)}
        assert float(mn.logL(x)) == float(mr.logL(x))


def _point(params, seed):
    rng = np.random.default_rng(seed)
    return {p: float(rng.uniform(-0.9, 0.9) if '_ecorr_corr_' in p else rng.uniform(-7.5, -5.5)) for p in params}


@pytest.mark.parametrize("nmodes", [2, 3])
def test_correlated_with_zero_correlations_is_the_independent_model(nmodes):
    psr = _read("multi_backend_pulsar")
    cor = _model(psr, signals.makegp_decorr(psr, nmodes=nmodes, correlated=True))
    ind = _model(psr, signals.makegp_decorr(psr, nmodes=nmodes))
    for seed in range(5):
        x = _point(ind.logL.params, seed)
        z = dict(x, **{p: 0.0 for p in cor.logL.params if '_ecorr_corr_' in p})
        assert float(cor.logL(z)) == pytest.approx(float(ind.logL(x)), rel=1e-12, abs=0)


def test_correlated_one_mode_is_makegp_ecorr():
    psr = _read("multi_backend_pulsar")
    cor = _model(psr, signals.makegp_decorr(psr, nmodes=1, correlated=True))
    plain = _model(psr, signals.makegp_ecorr(psr))
    assert cor.logL.params == plain.logL.params
    x = _point(plain.logL.params, 0)
    assert float(cor.logL(x)) == float(plain.logL(x))


def _cvine(rho, k):
    """R = L L^T from partial correlations rho[(a, b)], a > b, written out from the definition."""
    L = np.zeros((k, k))
    for a in range(k):
        for b in range(a):
            L[a, b] = rho[(a, b)] * np.sqrt(1.0 - np.sum(L[a, :b] ** 2))
        L[a, a] = np.sqrt(1.0 - np.sum(L[a, :a] ** 2))
    return L @ L.T


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("enterprise", [False, True])
def test_correlated_prior_is_the_identity_kron_d_r_d(name, enterprise):
    psr = _read(name)
    k = 3
    gp = signals.makegp_decorr(psr, nmodes=k, correlated=True, enterprise=enterprise)
    sizes = [len(c) for c in _ecorr_columns(psr, enterprise).values()]
    x = _point(gp.Phi.params, 1)
    phi = np.asarray(gp.Phi.getN(x))
    off = 0
    for lab, nep in zip(sorted(set(np.asarray(psr.backend_flags).tolist())), sizes):
        d = np.array([10 ** x[f'{psr.name}_{lab}_log10_ecorr' + ('' if m == 0 else f'_k{m}')]
                      for m in range(k)])
        rho = {(a, b): x[f'{psr.name}_{lab}_ecorr_corr_k{a}k{b}'] for a in range(1, k) for b in range(a)}
        M = np.outer(d, d) * _cvine(rho, k)
        block = phi[off:off + nep * k, off:off + nep * k]
        assert np.allclose(block, np.kron(np.eye(nep), M), rtol=1e-12, atol=0)
        off += nep * k
    assert off == phi.shape[0]


def test_parameter_names_run_over_modes_then_pairs():
    psr = _read("multi_backend_pulsar")
    labs = sorted(b for b in set(np.asarray(psr.backend_flags).tolist()) if b != '')
    gp = signals.makegp_decorr(psr, nmodes=3, correlated=True)
    want = ([f'{psr.name}_{b}_log10_ecorr{m}' for b in labs for m in ('', '_k1', '_k2')]
            + [f'{psr.name}_{b}_ecorr_corr_k{a}k{c}' for b in labs for a, c in ((1, 0), (2, 0), (2, 1))])
    assert list(gp.Phi.params) == want


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("variable", [False, True])
def test_fixed_values_give_the_free_likelihood(correlated, variable):
    psr = _read("multi_backend_pulsar")
    free = _model(psr, signals.makegp_decorr(psr, nmodes=3, correlated=correlated))
    x = _point(free.logL.params, 4)
    gp = signals.makegp_decorr(psr, noisedict=x, nmodes=3, correlated=correlated, variable=variable)
    assert isinstance(gp, matrix.VariableGP if variable else matrix.ConstantGP)
    fixed = _model(psr, gp)
    assert fixed.logL.params == []
    assert float(fixed.logL({})) == pytest.approx(float(free.logL(x)), rel=1e-12, abs=0)


def _ecorr_columns(psr, enterprise=False):
    """The epoch-indicator columns of makegp_ecorr, grouped by backend label."""
    U = np.asarray(signals.makegp_ecorr(psr, enterprise=enterprise).F)
    labs = np.asarray(psr.backend_flags)
    out = {b: [] for b in sorted(b for b in set(labs.tolist()) if b != '')}
    for j in range(U.shape[1]):
        out[labs[U[:, j] != 0][0]].append(U[:, j].astype(np.float64))
    return out


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("enterprise", [False, True])
def test_basis_is_epoch_indicators_times_legendre_polynomials(name, enterprise):
    """Column (backend, epoch, m) is the epoch indicator times P_m(y), y in [-1, 1] per backend."""
    psr = _read(name)
    k = 3
    lnf = np.log(np.asarray(psr.freqs, dtype=np.float64))
    labs = np.asarray(psr.backend_flags)
    cols = []
    for b, epochs in _ecorr_columns(psr, enterprise).items():
        lo, hi = lnf[labs == b].min(), lnf[labs == b].max()
        y = np.zeros_like(lnf) if hi == lo else (lnf - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
        for u in epochs:
            cols += [u * np.polynomial.legendre.Legendre.basis(m)(y) for m in range(k)]
    want = np.column_stack(cols) if cols else np.zeros((len(lnf), 0))
    F = np.asarray(signals.makegp_decorr(psr, nmodes=k, enterprise=enterprise).F)
    assert F.shape == want.shape
    assert np.allclose(F, want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("correlated", [False, True])
def test_scale_multiplies_every_sigma(correlated):
    psr = _read("empty_epoch_pulsar")
    one = signals.makegp_decorr(psr, nmodes=3, correlated=correlated)
    two = signals.makegp_decorr(psr, nmodes=3, correlated=correlated, scale=2.0)
    x = _point(one.Phi.params, 5)
    assert np.allclose(np.asarray(two.Phi.getN(x)), 4.0 * np.asarray(one.Phi.getN(x)), rtol=1e-12, atol=0)


def test_a_bad_frequency_stays_in_its_backend():
    psr = _read("multi_backend_pulsar")
    labs = np.asarray(psr.backend_flags)
    first = sorted(set(labs.tolist()))[0]
    psr.freqs = np.asarray(psr.freqs, dtype=np.float64).copy()
    psr.freqs[np.nonzero(labs == first)[0][0]] = np.nan
    with np.errstate(invalid='ignore'):
        F = np.asarray(signals.makegp_decorr(psr, nmodes=3).F)
    assert np.all(np.isfinite(F[labs != first]))
    assert not np.all(np.isfinite(F))


def test_nmodes_takes_any_integer_and_nothing_else():
    psr = _read("single_backend_pulsar")
    a = signals.makegp_decorr(psr, nmodes=np.int64(2))
    assert np.array_equal(np.asarray(a.F), np.asarray(signals.makegp_decorr(psr, nmodes=2).F))
    for bad in (True, 0, 2.0):
        with pytest.raises(ValueError, match='nmodes'):
            signals.makegp_decorr(psr, nmodes=bad)
