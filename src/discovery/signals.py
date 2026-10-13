import os
import re
import inspect
import types
import typing
import warnings
from collections.abc import Iterable

import numpy as np
import scipy.interpolate as si
import jax
import jax.numpy as jnp

from . import matrix
from . import const
from . import solar

# residuals

def residuals(psr):
    return psr.residuals


# EFAC/EQUAD/ECORR noise

# no backends
def makenoise_measurement_simple(psr, noisedict={}, add_equad=True, tnequad=False):
    """Single-EFAC (optionally single-EQUAD) white-noise model for a pulsar.

    Builds a diagonal measurement-noise matrix using one ``efac`` for the whole
    pulsar. When ``add_equad`` is True an EQUAD term is included: ``tnequad=True``
    uses the TempoNest convention (EQUAD added outside the EFAC scaling), while the
    default (``tnequad=False``) uses the tempo2/``t2equad`` convention (EQUAD added
    in quadrature with the TOA errors, inside the EFAC scaling). Set
    ``add_equad=False`` for an EFAC-only model. If all required parameters are present
    in ``noisedict`` a constant matrix is returned, otherwise a variable one.
    """
    efac = f'{psr.name}_efac'
    if tnequad and add_equad:
        log10_tnequad = f'{psr.name}_log10_tnequad'
        params = [efac, log10_tnequad]
    elif add_equad:
        log10_t2equad = f'{psr.name}_log10_t2equad'
        params = [efac, log10_t2equad]
    else:
        params = [efac]

    if all(par in noisedict for par in params):
        if tnequad and add_equad:
            noise = noisedict[efac]**2 * psr.toaerrs**2 + (10.0**(2.0 * noisedict[log10_tnequad]))
        elif add_equad:
            noise = noisedict[efac]**2 * (psr.toaerrs**2 + 10.0**(2.0 * noisedict[log10_t2equad]))
        else:
            noise = noisedict[efac]**2 * psr.toaerrs**2
        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs = matrix.jnparray(psr.toaerrs)
        def getnoise(params, tnequad=tnequad):
            if tnequad and add_equad:
                return params[efac]**2 * toaerrs**2 + 10.0**(2.0 * params[log10_tnequad])
            elif add_equad:
                return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))
            else:
                return params[efac]**2 * toaerrs**2
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)


# nanograv backends
def selection_backend_flags(psr):
    return psr.backend_flags


def selection_flags(flags, sep='_', warn_below=10):
    """Return a selection splitting the TOAs on one or more per-TOA flags.

    Functions taking a selection, such as :func:`makenoise_measurement` and
    :func:`makegp_ecorr`, split on the backend by default; this splits on any entry of
    ``psr.flags`` instead, so a per-channel efac and equad is ``selection_flags('chan')``.
    Labels carry the flag name, giving parameters like ``J0437-4715_chan15_efac``.

    Raises if a named flag is absent, or if any TOA carries an empty value for one.

    flags:      flag name, or a sequence of names whose values are combined into one
                label
    sep:        separator between the parts of a combined label
    warn_below: warn about groups holding fewer than this many TOAs
    """
    names = [flags] if isinstance(flags, str) else list(flags)

    def selection(psr):
        missing = [n for n in names if n not in psr.flags]
        if missing:
            raise KeyError(f'selection_flags: {psr.name} has no flag(s) {missing}; '
                           f'available flags are {sorted(psr.flags)}.')

        cols = []
        for n in names:
            v = np.asarray(psr.flags[n]).astype(str)
            nempty = int((v == '').sum())
            if nempty:
                raise ValueError(
                    f'selection_flags: {psr.name} has {nempty} TOAs with an empty {n!r} '
                    f'flag. Those TOAs would be dropped from the selection and left with '
                    f'zero measurement noise, so fix the flag rather than proceeding.')
            cols.append(np.char.add(n, v))

        labels = cols[0] if len(cols) == 1 else np.array(
            [sep.join(parts) for parts in zip(*cols)])

        counts = {lab: int((labels == lab).sum()) for lab in set(labels.tolist())}
        thin = {k: v for k, v in counts.items() if v < warn_below}
        if thin:
            warnings.warn(f'selection_flags on {names} for {psr.name} gives '
                          f'{len(counts)} groups, of which {len(thin)} hold fewer than '
                          f'{warn_below} TOAs: {dict(sorted(thin.items())[:6])}'
                          f'{" ..." if len(thin) > 6 else ""}')

        return labels

    selection.__name__ = 'selection_' + '_'.join(names)
    selection.flags = names

    return selection


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, ecorr=False, selection=selection_backend_flags, vectorize=True,
                          outliers=False, enterprise=False):
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    if tnequad:
        log10_tnequads = [f'{psr.name}_{backend}_log10_tnequad' for backend in backends]
        params = efacs + log10_tnequads
    else:
        log10_t2equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]
        params = efacs + log10_t2equads

    masks = [(backend_flags == backend) for backend in backends]
    logscale = np.log10(scale)

    # scale each toa individually. register scales as a parameter
    if outliers:
        toaerr_scaling = f'{psr.name}_alpha_scaling({psr.toas.size})'
        params.append(toaerr_scaling)

    if all(par in noisedict for par in params):
        if outliers:
            raise ValueError("No outlier scaling if white noise is fixed.")
        if tnequad:
            noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        if ecorr:
            egp = makegp_ecorr(psr, noisedict=noisedict, enterprise=enterprise, scale=scale, selection=selection)
            return matrix.NoiseMatrixSM_novar(noise, egp.F, egp.Phi.N)
        else:
            return matrix.NoiseMatrix1D_novar(noise)
    else:
        if vectorize:
            toaerrs2, masks = matrix.jnparray(scale**2 * psr.toaerrs**2), matrix.jnparray([mask for mask in masks])

            if tnequad:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_tnequad])) for log10_tnequad in log10_tnequads])

                    return (masks * (efac2[:,jnp.newaxis] * (alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
            else:

                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_t2equad])) for log10_t2equad in log10_t2equads])

                    return (masks * efac2[:,jnp.newaxis] * ((alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
        else:
            toaerrs, masks = matrix.jnparray(scale * psr.toaerrs), [matrix.jnparray(mask) for mask in masks]
            if tnequad:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0

                    return sum(mask * (params[efac]**2 * (alpha_scaling * toaerrs**2) + 10.0**(2 * (logscale + params[log10_tnequad])))
                               for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
            else:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    return sum(mask * params[efac]**2 * (alpha_scaling * toaerrs**2 + 10.0**(2 * (logscale + params[log10_t2equad])))
                               for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        getnoise.params = params

        if ecorr:
            egp = makegp_ecorr(psr, noisedict={}, enterprise=enterprise, scale=scale, selection=selection)
            return matrix.NoiseMatrixSM_var(getnoise, egp.F, egp.Phi.getN)
        else:
            return matrix.NoiseMatrix1D_var(getnoise)


# ECORR

# quantization
# note the resulting ecorr degrees of freedom are slightly different than in enterprise
# (and of course I forgot about it)

# bins = (psr.toas + 0.5).astype(np.int64)
# uniques, counts = np.unique(bins, return_counts=True)
# Umat = jnp.array(np.vstack([bins == unique for unique, count in zip(uniques, counts) if count > 1]).astype(jnp.float64).T)

def quantize(toas, dt=1.0):
    isort = np.argsort(toas)
    bins = np.zeros_like(toas, np.int64)

    b, v = 0, toas.min()
    for j in isort:
        if toas[j] - v > dt:
            v = toas[j]
            b = b + 1

        bins[j] = b

    return bins

# no backends
def makegp_ecorr_simple(psr, noisedict={}):
    log10_ecorr = f'{psr.name}_log10_ecorr'
    params = [log10_ecorr]

    bins = quantize(psr.toas)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T
    ones = np.ones(Umat.shape[1], dtype=np.float64)

    if all(par in noisedict for par in params):
        phi = (10.0**(2.0 * noisedict[log10_ecorr])) * ones

        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umat)
    else:
        ones = matrix.jnparray(ones)
        def getphi(params):
            return (10.0**(2.0 * params[log10_ecorr])) * ones
        getphi.params = params

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umat)

# nanograv backends
def makegp_ecorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=selection_backend_flags, variable=False, name='ecorrGP'):
    """Per-backend ECORR: one jitter amplitude per epoch, shared by all its TOAs.

    This is :func:`makegp_decorr` with ``nmodes=1``, sampling ``{psr}_{backend}_log10_ecorr``;
    the parameters are those of :func:`makegp_decorr`.
    """
    return makegp_decorr(psr, noisedict=noisedict, enterprise=enterprise, scale=scale,
                         selection=selection, variable=variable, nmodes=1, name=name)


def makegp_decorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=selection_backend_flags, variable=False,
                  nmodes=1, correlated=False, name='decorrGP'):
    """Frequency-resolved ECORR: per-epoch jitter as a low-order Legendre series in log frequency.

    Within each backend the TOA frequencies are mapped to
    ``y = (ln nu - (ln nu_max + ln nu_min) / 2) / ((ln nu_max - ln nu_min) / 2)`` in [-1, 1],
    with nu_min and nu_max the backend's lowest and highest TOA frequencies. Epoch e of the
    backend carries the delay ``sum_m j_em P_m(y)``, m = 0 .. nmodes - 1, so the basis column
    for (e, m) is the epoch indicator times P_m(y). Columns are ordered by backend, then
    epoch, then mode.

    With ``correlated=False`` the amplitudes are independent, ``j_em ~ N(0, sigma_m^2)``. With
    ``correlated=True`` each epoch's amplitude vector has covariance ``M = D R D``, the same
    for all epochs of a backend, where ``D = diag(sigma_0, ..., sigma_{nmodes-1})`` and R is
    the correlation matrix built from partial correlations rho_ab by the canonical-vine
    Cholesky construction (rho_a0 the correlation of modes a and 0, rho_ab for b > 0 the
    partial correlation given modes 0 .. b-1). The prior of a backend's coefficients is then
    ``I_epoch (x) M``.
    All rho_ab = 0 gives the independent model, and ``nmodes=1`` is :func:`makegp_ecorr`, whose
    default ``name`` is ``'ecorrGP'``. A backend whose TOAs share one frequency has y = 0
    throughout.

    psr:        Pulsar
    noisedict:  fixed parameter values; all of the GP's parameters must be present to fix them
    enterprise: count only epochs with more than one TOA, as enterprise does
    scale:      multiplies every sigma_m
    selection:  callable psr -> per-TOA labels, one set of parameters per label
    variable:   with every parameter fixed, return a VariableGP rather than a ConstantGP
    nmodes:     number of Legendre modes, an integer >= 1
    correlated: give the modes the full covariance M rather than independent amplitudes
    name:       GP name, used in the coefficient index ``{psr.name}_{name}_coefficients(N)``

    Parameters, for each label ``lab``: ``{psr.name}_{lab}_log10_ecorr`` (log10 sigma_0),
    ``{psr.name}_{lab}_log10_ecorr_k{m}`` (log10 sigma_m, m >= 1) and, with
    ``correlated=True``, ``{psr.name}_{lab}_ecorr_corr_k{a}k{b}`` (rho_ab, a > b), in [-1, 1].

    Example::

        gp = makegp_decorr(psr, nmodes=5, correlated=True)
        model = PulsarLikelihood([psr.residuals, makegp_timing(psr, svd=True),
                                  makenoise_measurement(psr), gp])
    """
    if isinstance(nmodes, bool) or not isinstance(nmodes, (int, np.integer)) or nmodes < 1:
        raise ValueError('makegp_decorr: nmodes must be an integer >= 1')

    k = nmodes = int(nmodes)
    freqs = np.asarray(psr.freqs) if k > 1 else None
    log10_ecorrs, corrs, Umats = [], [], []

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append([f'{psr.name}_{backend}_log10_ecorr' + ('' if m == 0 else f'_k{m}')
                             for m in range(k)])
        corrs.append({(a, b): f'{psr.name}_{backend}_ecorr_corr_k{a}k{b}'
                      for a in range(1, k) for b in range(a)})

        # For handling the single backend case
        if len(np.unique(masks)) == 1:
            # for those pulsar with only one backend
            first_valid_bin = 0
        else:
            # if the mask contains zeros
            # the zeros in quantize below end up in the
            # first entry, which we skip later.
            first_valid_bin = 1

        bins = quantize(psr.toas * mask)

        if enterprise:
            # legacy accounting of degrees of freedom
            uniques, counts = np.unique(bins, return_counts=True)
            epoch_masks = [bins == i for i, cnt in zip(
                uniques[first_valid_bin:],
                counts[first_valid_bin:]) if cnt > 1]

            if epoch_masks:
                U_backend = np.vstack(epoch_masks).T
            else:
                # if there is no ToAs observed at the same time
                U_backend = np.zeros((len(bins), 0))

            Umats.append(U_backend)
        else:
            Umats.append(np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T)

        if k > 1:
            # per-backend reference frequency (controls the centring of x)
            fref_b = float(np.mean(freqs[mask]))
            x = np.log(freqs / fref_b)

            # rescale x linearly to y in [-1, 1] across this backend's TOAs
            x_backend = x[mask]
            x_min, x_max = float(x_backend.min()), float(x_backend.max())
            half_range = 0.5 * (x_max - x_min)
            if half_range == 0.0:
                y = np.zeros_like(freqs)
            else:
                y = (x - 0.5 * (x_max + x_min)) / half_range
            y = np.where(mask, y, 0.0)

            # columns of legvander are P_0(y) ... P_{nmodes-1}(y)
            P = np.polynomial.legendre.legvander(y, nmodes - 1)

            U0 = Umats[-1].astype(np.float64)
            nep = U0.shape[1]

            # columns ordered (epoch e outer, mode o inner) so the per-backend Phi
            # block is exactly kron(I_nep, M) below
            cols = np.empty((U0.shape[0], nep * k), dtype=np.float64)
            for e in range(nep):
                for o in range(k):
                    cols[:, e * k + o] = U0[:, e] * P[:, o]
            Umats[-1] = cols
    Umatall = np.hstack(Umats)
    amplitudes = [a for amps in log10_ecorrs for a in amps]
    params = amplitudes

    pmasks, cnt = [], 0
    for Umat in Umats:
        for m in range(k):
            z = np.zeros(Umatall.shape[1], dtype=np.float64)
            z[cnt+m:cnt+Umat.shape[1]:k] = 1.0
            pmasks.append(z)
        cnt = cnt + Umat.shape[1]
    logscale = np.log10(scale)

    if correlated and k > 1:
        params = []
        for amps in log10_ecorrs:
            params.extend(amps)
        for corr in corrs:
            params.extend(corr.values())

        jnp_ = matrix.jnp

        def build_M(p):
            mblocks = []
            for amps, corr in zip(log10_ecorrs, corrs):
                d = jnp_.stack([10.0 ** (logscale + p[a]) for a in amps])  # (k,) marginal std per mode
                # partial-correlation (C-vine) Cholesky factor L of the mode
                # correlation matrix R = L L^T. The parameters p[corr[(i, j)]] in
                # (-1, 1) are correlations (j == 0) / partial correlations (j > 0),
                # so R -- and hence M -- is PSD with unit diagonal for any values.
                L_rows = []
                for i in range(k):
                    row = [0.0] * k
                    acc = 0.0  # sum of squares of the entries already placed in this row
                    for j in range(i):
                        Lij = p[corr[(i, j)]] * jnp_.sqrt(jnp_.clip(1.0 - acc, 1e-12, None))
                        row[j] = Lij
                        acc = acc + Lij ** 2
                    row[i] = jnp_.sqrt(jnp_.clip(1.0 - acc, 1e-12, None))  # unit row norm
                    L_rows.append(jnp_.stack([jnp_.asarray(v, dtype=jnp_.float64) for v in row]))
                L = jnp_.stack(L_rows)                                 # (k, k) lower-triangular
                R = L @ L.T
                M = (d[:, None] * d[None, :]) * R                      # (k, k) PSD covariance
                mblocks.append(M)
            return mblocks
        nepochs = [U.shape[1] // k for U in Umats]

        def build_phi(p):
            return matrix.jsp.linalg.block_diag(*[jnp_.kron(jnp_.eye(n), M) for n, M in zip(nepochs, build_M(p))])

        if all(par in noisedict for par in params):
            phi = np.asarray(build_phi(noisedict), dtype=np.float64)

            if variable:
                def getphi(params):
                    return phi
                getphi.params = []

                gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umatall)
                gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
                gp.name, gp.pos = psr.name, psr.pos
                gp.gpname, gp.gpcommon = name, []

                return gp
            else:
                return matrix.ConstantGP(matrix.NoiseMatrix2D_novar(phi), Umatall)
        else:
            def getphi(params):
                return build_phi(params)
            getphi.params = params

            gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp

    if all(par in noisedict for par in params):
        phi = sum(10.0**(2 * (logscale + noisedict[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(amplitudes, pmasks))

        if variable:
            def getphi(params):
                return phi
            getphi.params = []

            gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umatall)
    else:
        pmasks = [matrix.jnparray(pmask) for pmask in pmasks]
        def getphi(params):
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(amplitudes, pmasks))
        getphi.params = params

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp


# timing model

def makegp_improper(psr, fmat, constant=1.0e40, name='improperGP', variable=False):
    if variable:
        phi = matrix.jnparray(constant * np.ones(fmat.shape[1]))

        def getphi(params):
            return phi
        getphi.params = []

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    else:
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)

    gp.name = psr.name
    gp.gpname = name

    return gp

def makegp_timing(psr, constant=None, variance=None, svd=False, scale=1.0, variable=False):
    if svd:
        fmat, _, _ = np.linalg.svd(scale * psr.Mmat, full_matrices=False)
    else:
        fmat = np.array(psr.Mmat / np.sqrt(np.sum(psr.Mmat**2, axis=0)), dtype=np.float64)

    if variance is None:
        if constant is None:
            constant = 1.0e40
        # else constant can stay what it is
    else:
        if constant is None:
            constant = variance * psr.Mmat.shape[0] / psr.Mmat.shape[1]
            return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable)
        else:
            raise ValueError("signals.makegp_timing() can take a specification of _either_ `constant` or `variance`.")

    return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable)


# static frequency-dependent delay

def normalise_tm_basis(psr, scale=1.0):
    """Timing-model design matrix with unit-norm columns.

    All-zero columns, as from a fitted parameter with no TOAs behind it, are dropped with a
    warning naming them.

    psr:   Pulsar with ``Mmat`` and, optionally, ``fitpars``
    scale: multiplies ``psr.Mmat`` before the columns are normalised
    """
    Mmat = np.asarray(scale * psr.Mmat, dtype=np.float64)
    norms = np.sqrt(np.sum(Mmat**2, axis=0))
    keep = norms > 0

    ndrop = int((~keep).sum())
    if ndrop:
        idx = np.where(~keep)[0]
        names = (list(np.asarray(psr.fitpars)[idx]) if hasattr(psr, 'fitpars')
                 else list(idx))
        warnings.warn(f'{psr.name} has {ndrop} all-zero timing-model column(s), '
                      f'dropped: {names}')

    return Mmat[:, keep] / norms[keep]


class _FewBins(ValueError):
    """Raised by :func:`_fd_bins` when a selection of TOAs supports fewer than two bins."""


def makegp_fd_powerlaw(psr, bins='chan', selection=None, flo_pad=2.0, nf=128, jitter=1e-8,
                       name='fd_gp'):
    """Static frequency-dependent (FD) delay, one per frequency bin, with a power-law prior.

    A delay that is the same at every epoch and varies across the band, such as the mismatch
    between a template and the frequency evolution of the profile::

        n_FD = F_FD c,   (F_FD)_ik = 1[bin_i = k]

    with ``q_k`` the mean of ``ln(nu)`` over the TOAs in bin ``k`` (:func:`_fd_bins`). ``bins`` names
    a per-TOA flag whose values are the bins, one per channel for ``'chan'``, or is an integer ``N``
    for ``N`` bins of equal width in ``ln(nu)`` between the lowest and highest TOA frequency, empty
    bins dropped. Bins of equal width in ``ln(nu)`` do not line up with a channel grid linear in
    frequency; on channelised data, use the channel flag.

    ``selection``, a callable returning a per-TOA label such as the observing system, gives each
    label its own block of bins, for systems that observe the same band with different
    channelisations or templates. The bins, their guards and the band ``[f_lo, f_hi]`` then apply
    within each block; the blocks share ``sigma_FD`` and ``gamma_FD``, and ``K`` is block diagonal
    across them before the bin-space reduction. A block with fewer than two bins, a single
    frequency, or too few bins for its band (``f_hi <= f_lo``) gets no bins, with a warning.

    The unit-amplitude prior on ``c`` is a power law over ``f``, in cycles per unit ``ln(nu)``,
    band-limited to ``[f_lo, f_hi]``::

        K_kl = int f**-gamma_FD cos(2 pi f (q_k - q_l)) df / int f**-gamma_FD df

    evaluated as the exact covariance of a step spectrum over ``nf`` log-spaced bins
    (:func:`_fd_powerlaw_correlation`), with ``f_hi = 1/(2 x median spacing of the q_k)`` and
    ``f_lo = 1/(flo_pad x span of the q_k)``. ``gamma_FD = 0`` is close to an independent offset per
    bin where the bins are close to uniform in ``ln(nu)``; at the Nyquist frequency the correlation
    of neighbouring bins changes at first order with their spacing. ``gamma_FD = 2 nu + 1`` has the
    high-frequency slope of a Matern kernel of order ``nu``.

    The timing-model span is projected out of the basis, ``F_FD <- (I - Q Q^T) F_FD``, with ``Q`` the
    orthonormal factor of the QR decomposition of ``M`` after its columns are scaled to unit norm
    (:func:`normalise_tm_basis`). The bin-space directions of the timing-model columns that depend
    on frequency alone (the offset, each ``FDk``, a constant DM, or a column the bins reproduce
    exactly, such as a jump between receivers) are removed from the prior
    (:func:`_fd_node_projection`): ``c = B a``, with ``B`` an orthonormal basis of the remaining bin
    space, and ``K <- B^T K B`` plus ``jitter`` times its mean diagonal on the diagonal. The prior is
    normalised so that ``sigma_FD`` is the rms delay per TOA after both projections::

        Phi = N_TOA sigma_FD**2 K / tr(F_FD K F_FD^T)

    with ``F_FD`` the projected basis. Samples ``{psr}_{name}_log10_sigma`` (``log10 sigma_FD``, in
    seconds) and ``{psr}_{name}_gamma`` (``gamma_FD``).

    Raises if the flag is missing, if a TOA has an empty flag value, if two flag bins overlap in
    frequency, if fewer than two bin directions survive the projections, or if ``f_hi <= f_lo``.
    The GP carries ``project_basis``, ``F_FD`` after the timing-model projection and before the
    bin-space reduction, which :func:`makegp_pa` projects out; ``node_basis``, ``B``;
    ``node_removed``, the names of the removed timing-model columns; ``fd_nodes``, the bin
    frequencies ``exp(q_k)``, a dict by label with ``selection``; and ``fd_bands``, the
    ``(f_lo, f_hi)`` of each block.

    Example::

        fd = makegp_fd_powerlaw(psr, bins='chan')
        m = PulsarLikelihood([psr.residuals, makenoise_measurement(psr),
                              makegp_timing(psr, svd=True), fd])

    psr:     Discovery Pulsar object
    bins:    per-TOA flag whose distinct values are the bins, or an integer number of bins of
             equal width in ``ln(nu)``
    selection: None, or a callable returning a per-TOA label; each label gets its own bins
    flo_pad: outer scale ``f_lo`` as ``1/(flo_pad x span)``
    nf:      log-spaced bins in the step spectrum
    jitter:  diagonal added to the projected prior, relative to its mean diagonal
    name:    parameter-name stem; :data:`prior.priordict_standard` has boxes for the default only
    """
    if not (np.isfinite(flo_pad) and flo_pad > 0):
        raise ValueError(f"makegp_fd_powerlaw: flo_pad must be positive and finite, got {flo_pad!r}.")
    if isinstance(nf, bool) or not isinstance(nf, (int, np.integer)) or nf < 1:
        raise ValueError(f"makegp_fd_powerlaw: nf must be an integer >= 1, got {nf!r}.")
    if not (np.isfinite(jitter) and jitter >= 0):
        raise ValueError(f"makegp_fd_powerlaw: jitter must be non-negative and finite, got {jitter!r}.")

    if selection is None:
        fmat_raw, q = _fd_bins(psr, bins)
        blocks = [(None, q)]
    else:
        labels = np.asarray(selection(psr)).astype(str)
        mats, blocks = [], []
        for label in sorted(set(labels.tolist())):
            try:
                F_b, q_b = _fd_bins(psr, bins, sel=labels == label)
            except _FewBins as err:
                warnings.warn(f"makegp_fd_powerlaw: {psr.name}: selection {label!r} gets no FD bins: {err}")
                continue
            dq_b = np.diff(np.sort(q_b))
            if not float(flo_pad) * float(q_b.max() - q_b.min()) > 2.0 * float(np.median(dq_b[dq_b > 0])):
                warnings.warn(f"makegp_fd_powerlaw: {psr.name}: selection {label!r} gets no FD bins: its "
                              f"{len(q_b)} bins put the bin Nyquist frequency at or below the outer scale.")
                continue
            mats.append(F_b)
            blocks.append((label, q_b))
        if not mats:
            raise ValueError(f"makegp_fd_powerlaw: {psr.name}: no selection supports two or more bins.")
        fmat_raw, q = np.hstack(mats), np.concatenate([q_b for _, q_b in blocks])
    Q_tm, _ = np.linalg.qr(normalise_tm_basis(psr))
    fmat = fmat_raw - Q_tm @ (Q_tm.T @ fmat_raw)

    B, removed = _fd_node_projection(psr, fmat_raw, q)
    fmat_out = fmat @ B
    if fmat_out.shape[1] < 2:
        raise ValueError(f"makegp_fd_powerlaw: {psr.name}: the timing model spans all but "
                         f"{fmat_out.shape[1]} of the {len(q)} bin directions; at least two are needed.")

    bands, corrs = [], []
    for label, q_b in blocks:
        dq = np.diff(np.sort(q_b))
        f_hi = 1.0 / (2.0 * float(np.median(dq[dq > 0])))
        f_lo = 1.0 / (float(flo_pad) * float(q_b.max() - q_b.min()))
        if not f_hi > f_lo:
            raise ValueError(f"makegp_fd_powerlaw: {psr.name}: the bin Nyquist "
                             f"1/{1 / f_hi:.3g} is not above the outer scale 1/{1 / f_lo:.3g} "
                             f"(per unit log freq); lower flo_pad or use more bins.")
        bands.append((f_lo, f_hi))
        corrs.append(_fd_powerlaw_correlation(q_b, f_lo, f_hi, nf))

    Bj = matrix.jnparray(B)
    eye_out = matrix.jnparray(np.eye(B.shape[1]))

    signame, gamname = f'{psr.name}_{name}_log10_sigma', f'{psr.name}_{name}_gamma'

    def shape(params):
        K = (corrs[0](params[gamname]) if len(corrs) == 1 else
             jax.scipy.linalg.block_diag(*[c(params[gamname]) for c in corrs]))
        Ks = Bj.T @ K @ Bj
        return Ks + jitter * jnp.mean(jnp.diag(Ks)) * eye_out

    ntoa = fmat_out.shape[0]
    gram = matrix.jnparray(fmat_out.T @ fmat_out)

    def getphi(params):
        Ks = shape(params)
        g = jnp.sum(gram * Ks) / ntoa
        return 10.0 ** (2.0 * params[signame]) / g * Ks
    getphi.params = [signame, gamname]

    Phi = matrix.NoiseMatrix2D_var(getphi)

    gp = matrix.VariableGP(Phi, fmat_out)
    gp.index = {f'{psr.name}_{name}_coefficients({fmat_out.shape[1]})': slice(0, fmat_out.shape[1])}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []
    gp.fd_nodes = np.exp(q) if selection is None else {label: np.exp(q_b) for label, q_b in blocks}
    gp.node_basis, gp.node_removed, gp.fd_bands = B, removed, bands
    gp.project_basis = fmat

    return gp


def _fd_bins(psr, bins, what='makegp_fd_powerlaw', sel=None):
    """Indicator basis over frequency bins, and the bin positions.

    Returns ``(F, q)``: ``F[i, k] = 1`` if TOA ``i`` is in bin ``k`` and 0 otherwise, so a coefficient
    is a delay in seconds, and ``q_k`` the mean of ``ln(nu)`` over the TOAs in bin ``k``. A string
    ``bins`` names a per-TOA flag, one bin per value, in numeric order where every value is an
    integer; an integer gives that many bins of equal width in ``ln(nu)`` between the lowest and
    highest TOA frequency, empty bins dropped; where the TOA frequencies form clusters (the
    gaps between clusters more than 100 times the median gap), an edge that falls inside a
    cluster is warned about. Raises for non-finite or non-positive frequencies, a single frequency, a
    missing flag, an empty flag value, flag bins that overlap in frequency (the highest TOA
    frequency of one bin above the lowest of the next), or fewer than two non-empty bins.

    psr:  Discovery Pulsar object
    bins: flag name, or integer number of bins
    what: caller named in messages
    """
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{what}: {psr.name}: psr.freqs must be positive and finite.")
    sel = np.ones(len(x), dtype=bool) if sel is None else np.asarray(sel, dtype=bool)
    xs = x[sel]
    if xs.max() == xs.min():
        raise _FewBins(f"{what}: {psr.name}: every TOA is at one frequency, so there is no band to bin.")
    if isinstance(bins, str):
        if bins not in psr.flags:
            raise KeyError(f"{what}: {psr.name} has no flag {bins!r}; available "
                           f"flags are {sorted(psr.flags)}.")

        labels = np.asarray(psr.flags[bins]).astype(str)
        present = set(labels[sel].tolist())
        if '' in present:
            raise ValueError(f"{what}: {psr.name}: {int((labels[sel] == '').sum())} TOAs carry an empty "
                             f"{bins!r} flag, so they would have no bin.")

        try:
            order = sorted(present, key=int)
        except ValueError:
            # not every label is an integer, so plain string order is the only option
            order = sorted(present)

        pairs = [(v, sel & (labels == v)) for v in order]
    elif isinstance(bins, (int, np.integer)) and not isinstance(bins, bool) and bins >= 2:
        edges = np.linspace(xs.min(), xs.max(), int(bins) + 1)[1:-1]
        which = np.searchsorted(np.asarray(edges, dtype=np.float64), x)
        pairs = [(str(b), sel & (which == b)) for b in range(len(edges) + 1)]
    else:
        raise ValueError(f"{what}: bins must be a flag name or an integer >= 2, got {bins!r}.")

    counts = np.array([int(m.sum()) for _, m in pairs])
    pairs = [(lab, m) for (lab, m), n in zip(pairs, counts) if n > 0]

    if len(pairs) < 2:
        raise _FewBins(f"{what}: {psr.name}: bins={bins!r} gives {len(pairs)} non-empty bin(s); "
                         f"at least two are needed.")

    if isinstance(bins, str):
        stats = sorted((x[m].mean(), x[m].min(), x[m].max(), lab) for lab, m in pairs)
        clash = [(a[3], b[3]) for a, b in zip(stats[:-1], stats[1:]) if a[2] > b[1]]
        if clash:
            raise ValueError(
                f"{what}: {psr.name}: {len(clash)} pair(s) of {bins!r} bins overlap in frequency, "
                f"e.g. values {clash[0][0]!r} and {clash[0][1]!r}; a single TOA can cause this. A channel "
                f"index reused across receivers does this; combine it with the receiver flag into a "
                f"new flag (e.g. psr.flags['rcvr_chan'] = np.char.add(np.char.add(rcvr, '_'), chan)) "
                f"and pass that. Systems that observe the same band need selection=, which gives "
                f"each its own bins.")

    if not isinstance(bins, str):
        # count the edges that fall inside a populated frequency block rather than in a
        # gap between blocks
        uf = np.unique(np.exp(xs))
        gaps = np.diff(uf)
        nb = len(pairs)
        if len(gaps) >= nb > 1 and float(np.sort(gaps)[-(nb - 1)]) > 100.0 * float(np.median(gaps)):
            boundary = float(np.sort(gaps)[-(nb - 1)])
            j = np.clip(np.searchsorted(uf, np.exp(edges)), 1, len(uf) - 1)
            split = int(np.sum((uf[j] - uf[j - 1]) < 0.5 * boundary))
            if split:
                warnings.warn(f"{what}: {psr.name}: bins={bins} puts {split} of {len(edges)} edges "
                              f"inside a populated frequency block, splitting it across two bins; "
                              f"a channel flag places the bins on the channels.")
        live = np.array([int(m.sum()) for _, m in pairs])
        thin = int(np.sum(live < 0.5 * np.median(live)))
        if thin:
            warnings.warn(f"{what}: {psr.name}: {thin} of {len(live)} "
                          f"bins hold below half the median TOA count ({int(np.median(live))}).")

    fmat = np.zeros((len(x), len(pairs)), dtype=np.float64)
    q = np.empty(len(pairs), dtype=np.float64)
    for i, (_, mask) in enumerate(pairs):
        fmat[mask, i] = 1.0
        q[i] = float(np.mean(x[mask]))

    return fmat, q


def _fd_node_projection(psr, F, q, fit_tol=1e-9, degree=12, tol=1e-10):
    """Orthonormal basis of the bin space left once the timing model's frequency-only
    directions are removed, and the names of the columns they come from.

    A column of ``psr.Mmat`` counts when its per-TOA values are reproduced to relative residual
    ``fit_tol`` either by a function of frequency in the span of Legendre polynomials in ``ln(nu)``
    up to ``degree`` and ``nu**-2`` (the offset, every ``FDk``, a constant DM), evaluated at ``q``, or
    by the bin basis ``F`` itself (a jump between groups of bins), taken as its least-squares bin
    representation. The patterns are orthonormalised with singular values below ``tol`` of the
    largest dropped, so a prior ``B^T K B`` on the returned ``B`` puts no power along them.

    psr: Discovery Pulsar object with ``Mmat`` and, optionally, ``fitpars``
    F:   indicator basis before any projection
    q:   bin positions in ``ln(nu)``
    """
    M = np.asarray(psr.Mmat, dtype=np.float64)
    x = np.log(np.asarray(psr.freqs, dtype=np.float64))
    lo, hi = float(x.min()), float(x.max())

    def functions(y):
        u = (2.0 * y - lo - hi) / (hi - lo)
        return np.column_stack([np.polynomial.legendre.legvander(u, degree), np.exp(-2.0 * (y - lo))])

    at_toas, at_nodes = functions(x), functions(np.asarray(q, dtype=np.float64))
    names = [str(n) for n in getattr(psr, 'fitpars', [])]
    if len(names) != M.shape[1]:
        names = [f'Mmat column {j}' for j in range(M.shape[1])]

    patterns, labels = [], []
    for j in range(M.shape[1]):
        m = M[:, j]
        scale = np.linalg.norm(m)
        if scale == 0.0:
            continue
        coef, *_ = np.linalg.lstsq(at_toas, m, rcond=None)
        if np.linalg.norm(at_toas @ coef - m) < fit_tol * scale:
            patterns.append(at_nodes @ coef)
            labels.append(names[j])
            continue
        g, *_ = np.linalg.lstsq(F, m, rcond=None)
        if np.linalg.norm(F @ g - m) < fit_tol * scale:
            patterns.append(g)
            labels.append(names[j])

    n = len(q)
    if not patterns:
        warnings.warn(f'makegp_fd_powerlaw: {psr.name}: no timing-model column depends on frequency alone; '
                      f'node space left whole ({n} directions).')
        return np.eye(n), labels

    U, S, _ = np.linalg.svd(np.column_stack(patterns), full_matrices=True)
    r = int(np.sum(S > tol * S[0]))
    B = U[:, r:]
    return B, labels


def _fd_powerlaw_correlation(q, f_lo, f_hi, nf):
    """Band-limited power law in log-frequency as a correlation matrix over nodes ``q``.

    ``S(f) = f**-gamma`` over ``nf`` log-spaced bins of ``[f_lo, f_hi]``, each bin's weight
    ``w_j`` its Simpson integral and the bin taken flat across its width ``df_j``, so

        k(dq) = sum_j w_j cos(2 pi f_j dq) sinc(df_j dq),

    normalised to ``k(0) = 1``, is the exact covariance of that step spectrum. Returns
    ``corr(gamma)``.
    """
    edges = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), int(nf) + 1))
    a, b = matrix.jnparray(edges[:-1] / f_hi), matrix.jnparray(edges[1:] / f_hi)
    m = 0.5 * (a + b)
    dq = np.asarray(q)[None, :, None] - np.asarray(q)[None, None, :]
    cosines = matrix.jnparray(np.cos(2.0 * np.pi * f_hi * np.asarray(m)[:, None, None] * dq)
                              * np.sinc(f_hi * (np.asarray(b) - np.asarray(a))[:, None, None] * dq))

    def corr(gamma):
        w = (b - a) / 6.0 * (a ** -gamma + 4.0 * m ** -gamma + b ** -gamma)
        return jnp.tensordot(w / jnp.sum(w), cosines, axes=1)

    return corr


# parallactic angle

# ITRF geocentric coordinates in metres, verbatim from tempo2's observatories.dat
OBSERVATORY_ITRF = {
    'pks':     (-4554231.500,  2816759.100, -3454036.300),   # PARKES / MURRIYANG
    'meerkat': ( 5109360.133,  2006852.586, -3238948.127),   # MEERKAT
}

_WGS84_A = 6378137.0
_WGS84_E2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563)


def _itrf_to_geodetic(x, y, z):
    """Geodetic latitude and longitude, in radians, from ITRF metres on WGS84.

    The latitude is geodetic, i.e. measured from the local vertical.
    """
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h)))

    return lat, lon


def observatory_location(site):
    """Geodetic latitude and longitude in radians for an observing site.

    site: tempo2 observatory code, matched case-insensitively against
          :data:`OBSERVATORY_ITRF`, or an explicit ``(latitude, longitude)`` pair
          in degrees
    """
    if not isinstance(site, str):
        lat, lon = site
        return np.deg2rad(float(lat)), np.deg2rad(float(lon))

    key = site.strip().lower()
    xyz = OBSERVATORY_ITRF.get(key)
    if xyz is None:
        raise KeyError(
            f'observatory_location: no coordinates for site {site!r}. Known sites are '
            f'{sorted(OBSERVATORY_ITRF)}; for any other, pass '
            f'site=(latitude_deg, longitude_deg).')

    return _itrf_to_geodetic(*xyz)


def greenwich_sidereal_angle(mjd_utc):
    """Greenwich mean sidereal time in radians, from a UTC MJD.

    UT1 is approximated by UTC and the equation of the equinoxes is dropped, together
    worth under 0.01 degrees of hour angle.
    """
    d = np.asarray(mjd_utc, dtype=np.float64) + (2400000.5 - 2451545.0)
    T = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T**2 - T**3 / 38710000.0

    return np.deg2rad(np.mod(deg, 360.0))


def parallactic_angle(psr, site):
    """Parallactic angle at each TOA, in radians, in ``(-pi, pi]``.

    The angle between the great circle from the source to the zenith and the great
    circle from the source to the north celestial pole,

        tan(psi) = sin(H) cos(lat) / [sin(lat) cos(dec) - cos(lat) sin(dec) cos(H)]

    with ``H`` the local hour angle and ``lat`` the geodetic latitude of the site.

    Computed from ``psr.stoas``, the site arrival times in seconds, not from the barycentric
    ``psr.toas``. The source position is used as given, without precession or
    nutation to the date, and the sidereal time is the mean one
    (:func:`greenwich_sidereal_angle`).

    psr:  Discovery Pulsar object, carrying ``stoas`` and the source position in
          ``phi`` and ``theta``
    site: observing site, a code in :data:`OBSERVATORY_ITRF` or a ``(latitude, longitude)``
          pair in degrees
    """
    if not hasattr(psr, 'stoas'):
        raise AttributeError(
            f'parallactic_angle: {psr.name} has no stoas, the site arrival times. The '
            f'barycentric toas are not a substitute -- they carry the Roemer delay, '
            f'which is degrees of hour angle. Rebuild the feather with the stoas column.')

    mjd = np.asarray(psr.stoas, dtype=np.float64) / 86400.0
    ra, dec = float(psr.phi), 0.5 * np.pi - float(psr.theta)

    lat, lon = observatory_location(site)
    H = greenwich_sidereal_angle(mjd) + lon - ra

    return np.arctan2(np.sin(H) * np.cos(lat),
                      np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(H))


def makegp_pa(psr, site, bins='chan', selection=None, project=None, name='pa_gp'):
    """Parallactic-angle (PA) delay with a free amplitude and phase per frequency bin.

    The feeds of an altitude-azimuth telescope turn with the parallactic angle ``psi``, so
    polarisation leakage that the calibration leaves shifts the arrival time in a bin by
    ``A cos(2 psi - phi_0)``, with ``A`` and ``phi_0`` differing from bin to bin. Each bin therefore
    gets a coefficient on ``sin(2 psi)`` and one on ``cos(2 psi)``::

        n_PA = F_PA p,   F_PA = [ F_FD * sin(2 psi) | F_FD * cos(2 psi) ]

    where ``F_FD`` is the indicator basis of :func:`makegp_fd_powerlaw` over the same ``bins``, ``*``
    multiplies each column element-wise by the sinusoid at the TOAs (with ``selection``, the bins
    of each label in turn), and ``p`` holds the
    ``2 N_bin`` coefficients, the sin block then the cos block. ``psi`` is
    :func:`parallactic_angle` at ``site``.

    The prior is ``Phi = sigma_PA**2 I``, sampling ``{psr}_{name}_log10_sigma``, so that a bin of
    amplitude ``A`` has ``<A**2> = 2 sigma_PA**2`` and a uniform phase. The timing-model span is
    projected out of the basis, ``F_PA <- (I - Q Q^T) F_PA``, and ``Phi`` is left alone, so
    ``sigma_PA**2 F_PA F_PA^T`` is the prior that the unprojected model induces on what remains;
    columns the projection annihilates are kept. The basis can overlap a static per-bin delay:

    - ``project=None``: the PA basis keeps that static part, and can take up a static delay in
      each bin;
    - ``project=fd``, a GP from :func:`makegp_fd_powerlaw`: its ``project_basis`` is projected out
      as well, and only the part of the delay that varies with ``psi`` is left to this GP.

    ``gp.pa_rank`` is the rank of the projected basis (singular values above ``1e-8`` of the
    largest) and ``gp.pa_retained`` the fraction of the basis power, its squared Frobenius norm,
    that the projections leave.

    Example, with the static delay projected out::

        fd = makegp_fd_powerlaw(psr, bins='chan')
        pa = makegp_pa(psr, bins='chan', site='meerkat', project=fd)

    psr:     Discovery Pulsar object with ``stoas`` in seconds
    site:    observing site, a code in :data:`OBSERVATORY_ITRF` or a ``(latitude, longitude)``
             pair in degrees
    bins:    as :func:`makegp_fd_powerlaw`
    selection: as :func:`makegp_fd_powerlaw`; each label gets its own bins
    project: None, or a GP from :func:`makegp_fd_powerlaw`
    name:    parameter-name stem; :data:`prior.priordict_standard` has boxes for the default only
    """
    psi = parallactic_angle(psr, site=site)
    if selection is None:
        ind, q = _fd_bins(psr, bins, what='makegp_pa')
    else:
        labels = np.asarray(selection(psr)).astype(str)
        mats, qs = [], []
        for label in sorted(set(labels.tolist())):
            try:
                F_b, q_b = _fd_bins(psr, bins, what='makegp_pa', sel=labels == label)
            except _FewBins as err:
                warnings.warn(f"makegp_pa: {psr.name}: selection {label!r} gets no PA bins: {err}")
                continue
            mats.append(F_b)
            qs.append(q_b)
        if not mats:
            raise ValueError(f"makegp_pa: {psr.name}: no selection supports two or more bins.")
        ind, q = np.hstack(mats), np.concatenate(qs)

    Q_null = np.linalg.qr(normalise_tm_basis(psr))[0]
    if project is not None:
        P = getattr(project, 'project_basis', None)
        if P is None:
            raise TypeError(f'makegp_pa: project must be None or an FD GP from makegp_fd_powerlaw, '
                            f'which carries project_basis; got {type(project).__name__}.')
        P = np.asarray(P, dtype=np.float64)
        if P.shape[0] != len(psi) or getattr(project, 'name', psr.name) != psr.name:
            raise ValueError(f'makegp_pa: {psr.name}: project is an FD GP for '
                             f'{getattr(project, "name", "?")} with {P.shape[0]} TOAs, not this pulsar.')
        P = P - Q_null @ (Q_null.T @ P)
        Up, Sp, _ = np.linalg.svd(P, full_matrices=False)
        keep = Up[:, Sp > 1e-10 * Sp[0]]
        Q_null = np.hstack([Q_null, keep])

    fmat = np.hstack([ind * np.sin(2 * psi)[:, None],
                      ind * np.cos(2 * psi)[:, None]])

    before = np.linalg.svd(fmat, compute_uv=False)
    fmat = fmat - Q_null @ (Q_null.T @ fmat)
    after = np.linalg.svd(fmat, compute_uv=False)
    rank = int(np.sum(after > 1e-8 * after[0])) if after[0] > 0 else 0

    signame = f'{psr.name}_{name}_log10_sigma'
    ones = matrix.jnparray(np.ones(fmat.shape[1]))

    def getphi(params):
        return 10.0**(2.0 * params[signame]) * ones
    getphi.params = [signame]

    gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []
    gp.pa_bins = np.exp(q)
    gp.pa_angle = psi
    gp.pa_rank = rank
    gp.pa_retained = float(np.sum(after**2) / np.sum(before**2))

    return gp


# Fourier GP

def getspan(psrs):
    if isinstance(psrs, Iterable):
        return max(psr.toas.max() for psr in psrs) - min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.max() - psrs.toas.min()

def getstart(psrs):
    if isinstance(psrs, Iterable):
        return min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.min()


def fourierbasis(psr, components, T=None):
    if T is None:
        T = getspan(psr)

    f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*components), dtype=np.float64)
    for i in range(components):
        fmat[:, 2*i  ] = np.sin(2.0 * np.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * np.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def fourierbasis_dm(psr, components, T=None, fref=1400.0):
    """Fourier design matrix for a DM (dispersion measure) Gaussian process.

    Identical to :func:`fourierbasis`, but each row is scaled by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``, i.e. a fixed chromatic index
    alpha = 2. Use :func:`fourierbasis_chrom` when the chromatic index is a free
    parameter, in which case the process is general chromatic noise rather than DM.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    Dm = (fref / psr.freqs)**2

    return f, df, fmat * Dm[:, None]

def fourierbasis_chrom(psr, components, T=None, fref=1400.0):
    """Fourier design matrix for a chromatic Gaussian process with variable index.

    Returns a callable design-matrix factory ``fmatfunc(alpha)`` that scales the
    achromatic :func:`fourierbasis` columns by ``(fref / psr.freqs) ** alpha``,
    where the chromatic index ``alpha`` is a free parameter. Because alpha is not
    fixed to 2 the resulting process is general chromatic noise, not DM; use
    :func:`fourierbasis_dm` for the alpha = 2 (DM) case.
    """
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(alpha):
        return fmat * fnorm[:, None]**alpha

    return f, df, fmatfunc

def make_fourierbasis_dm(alpha=2.0, tndm=False):
    """Build a DM Fourier-basis function with a fixed chromatic index ``alpha``.

    Returns a ``basis(psr, components, T, fref)`` callable whose columns are the
    achromatic :func:`fourierbasis` scaled by ``(fref / psr.freqs) ** alpha``. With
    ``tndm=True`` the tempo2/TempoNest DM normalisation is also applied. A genuine
    DM basis should keep ``alpha = 2``; for a variable chromatic index use
    :func:`fourierbasis_chrom`.
    """
    def basis(psr, components, T=None, fref=1400.0):
        f, df, fmat = fourierbasis(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return f, df, fmat * Dm[:, None]

    return basis

def make_fourierbasis_chrom(alpha=4.0, tndm=False):
    """Build a chromatic Fourier-basis function with a fixed chromatic index ``alpha``.

    Thin wrapper around :func:`make_fourierbasis_dm` with a default ``alpha = 4`` (a
    common scattering-like index). The returned basis scales the achromatic
    :func:`fourierbasis` columns by ``(fref / psr.freqs) ** alpha``. Use this for a
    fixed-index chromatic process; for DM use :func:`make_fourierbasis_dm` (alpha = 2).
    """
    return make_fourierbasis_dm(alpha=alpha, tndm=tndm)

def dmfourierbasis(psr, components, T=None, fref=1400.0):
    warnings.warn("dmfourierbasis is deprecated; use fourierbasis_dm instead.",
                  DeprecationWarning, stacklevel=2)
    return fourierbasis_dm(psr, components, T=T, fref=fref)

def dmfourierbasis_alpha(psr, components, T=None, fref=1400.0):
    warnings.warn("dmfourierbasis_alpha is deprecated; use fourierbasis_chrom instead.",
                  DeprecationWarning, stacklevel=2)
    return fourierbasis_chrom(psr, components, T=T, fref=fref)

def dmfourierbasis_solar(psr, components, T=None):
    f, df, fmat = fourierbasis(psr, components, T)
    shape = solar.make_solardm(psr)(1.0)

    return f, df, fmat * shape[:, None]

def make_dmfourierbasis(alpha=2.0, tndm=False):
    warnings.warn("make_dmfourierbasis is deprecated; use make_fourierbasis_dm instead.",
                  DeprecationWarning, stacklevel=2)
    return make_fourierbasis_dm(alpha=alpha, tndm=tndm)

def makegp_fourier(psr, prior, components, T=None, mean=None, fourierbasis=fourierbasis, common=[], exclude=['f', 'df'], name='fourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    f, df, fmat = fourierbasis(psr, components, T)

    # f, df = matrix.jnparray(f), matrix.jnparray(df)
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap
    priorfunc.type = getattr(prior, 'type', None)

    if callable(fmat):
        argspec = inspect.getfullargspec(fmat)
        fargmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                   for arg in argspec.args if arg not in ['f', 'df']]

        def fmatfunc(params):
            return fmat(*[params[arg] for arg in fargmap])
        fmatfunc.params = fargmap

    gp = matrix.VariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmatfunc if callable(fmat) else fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(0,len(f))} # better for cosine
    gp.name, gp.pos = psr.name, psr.pos
    gp.gpname, gp.gpcommon = name, common

    if mean is not None:
        margspec = inspect.getfullargspec(mean)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]
        margmap = {arg: (arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
#                        won't work here since components already applies to frequencies
#                        + (f'({components})' if (margspec.annotations.get(arg) == typing.Sequence and components is not None) else '')
                   for arg in margs if not hasattr(psr, arg) and arg not in exclude}

        psrpars = {arg: getattr(psr, arg) for arg in margspec.args if hasattr(psr, arg)}

        def meanfunc(params):
            return mean(f, df, *psrpars.values(), **{arg: params[argname] for arg, argname in margmap.items()})
        meanfunc.params = sorted(margmap.values())

        gp.mean = meanfunc

    return gp


# for use in ArrayLikelihood. Same process for all pulsars.
def makecommongp_fourier(psrs, prior, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'], vector=False,
                         name='fourierCommonGP', meansname='meanFourierCommonGP'):
    argspec = inspect.getfullargspec(prior)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in exclude]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = fs[0], dfs[0]

    if vector:
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if f'({len(psrs)})' in arg else None for arg in argmap])

        def priorfunc(params):
            return vprior(f, df, *[params[arg] for arg in argmap])

        priorfunc.params = sorted(argmap)
        priorfunc.type = getattr(prior, 'type', None)
    else:
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if isinstance(argmap, list) else None for argmap in argmaps])

        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(f, df, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))
        priorfunc.type = getattr(prior, 'type', None)

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix12D_var(priorfunc), fmats)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(len(f)*i,len(f)*(i+1))
                for i, psr in enumerate(psrs)}

    if means is not None:
        margspec = inspect.getfullargspec(means)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]

        # parameters carried by the pulsar objects (e.g., pos), should be at the beginning of function
        psrpars = [{arg: getattr(psr, arg) for arg in margspec.args if hasattr(psrs[0], arg) and arg not in exclude}
                   for psr in psrs]

        # other means parameters, either common or pulsar-specific
        margmaps = [{arg: f'{meansname}_{arg}' if (f'{meansname}_{arg}' in common or arg in common) else f'{psr.name}_{meansname}_{arg}'
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude} for psr in psrs]

        def meanfunc(params):
            return matrix.jnparray([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp


# these support leave-one-out PPC

def makegp_fourier_delay(psr, components, T=None, name='fourierGP'):
    argname = f'{psr.name}_{name}_mean({components*2})'

    _, _, fmat = fourierbasis(psr, components, T)
    Fmat = matrix.jnparray(fmat)

    def delayfunc(params):
        return matrix.jnp.dot(Fmat, params[argname])
    delayfunc.params = [argname]

    return delayfunc

def makegp_fourier_variance(psr, components, T=None, name='fourierGP', noisedict={}):
    argname = f'{psr.name}_{name}_variance({components*2},{components*2})'

    _, _, fmat = fourierbasis(psr, components, T)

    if argname in noisedict:
        return matrix.ConstantGP(matrix.NoiseMatrix2D_novar(noisedict[argname]), fmat)
    else:
        def priorfunc(params):
            return params[argname]
        priorfunc.params = [argname]

        return matrix.VariableGP(matrix.NoiseMatrix2D_var(priorfunc), fmat)

# Global Fourier GP

# makes a block-diagonal GP over all pulsars; returns a GlobalVariableGP object in which
# the prior is the concatenation of single-pulsar priors; with common variables, it can be used
# to implement CURN as a globalgp object, or to set up the optimal statistic
def makegp_fourier_allpsr(psrs, prior, components, T=None, fourierbasis=fourierbasis, common=[], name='allpsrFourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                for arg in argspec.args if arg not in ['f', 'df']] for psr in psrs]

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])

    def priorfunc(params):
        return jnp.concatenate([prior(f, df, *[params[arg] for arg in argmap]) for argmap in argmaps])
    priorfunc.params = sorted(set(sum(argmaps, [])))

    def invprior(params):
        p = priorfunc(params)
        return 1.0 / p, jnp.sum(jnp.log(p))
    invprior.params = priorfunc.params

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix1D_var(priorfunc), fmats)
    gp.Phi_inv = invprior

    gp.index = {f'{psr.name}_{name}_coefficients({2*components})':
                slice((2*components)*i, (2*components)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp


def makeglobalgp_fourier(psrs, priors, orfs, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'],
                         name='fourierGlobalGP', meansname='meanFourierGlobalGP'):
    priors = priors if isinstance(priors, list) else [priors]
    orfs   = orfs   if isinstance(orfs, list)   else [orfs]

    argmaps = []
    for prior, orf in zip(priors, orfs):
        argspec = inspect.getfullargspec(prior)
        priorname = f'{name}' if len(priors) == 1 else f'{name}_{re.sub("_", "", orf.__name__)}'
        argmaps.append([f'{priorname}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                        for arg in argspec.args if arg not in exclude])

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])

    orfmats = [matrix.jnparray([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs]) for orf in orfs]

    if len(priors) == 1 and len(orfs) == 1:
        prior, orfmat, argmap = priors[0], orfmats[0], argmaps[0]

        def priorfunc(params):
            phi = prior(f, df, *[params[arg] for arg in argmap])

            # the jnp.dot handles the "pixel basis" case where the elements of orfmat are n-vectors
            # and phidiag is an (m x n)-matrix; here n is the number of pixels and m of Fourier components
            return jnp.block([[jnp.make2d(jnp.dot(phi, val)) for val in row] for row in orfmat])
        priorfunc.params = argmap
        priorfunc.type = jax.Array

        # if we're not in the pixel-basis case we can take a shortcut in making the inverse
        if orfmat.ndim == 2:
            invorf, orflogdet = matrix.jnparray(np.linalg.inv(orfmat)), np.linalg.slogdet(orfmat)[1]
            def invprior(params):
                phi = prior(f, df, *[params[arg] for arg in argmap])
                invphi = 1.0 / phi if phi.ndim == 1 else jnp.linalg.inv(phi)
                logdetphi = jnp.sum(jnp.log(phi)) if phi.ndim == 1 else jnp.linalg.slogdet(phi)[1]

                # |S_ij Gamma_ab| = prod_i (|S_i Gamma_ab|) = prod_i (S_i^npsr |Gamma_ab|)
                # log |S_ij Gamma_ab| = log (prod_i S_i^npsr) + log prod_i |Gamma_ab|
                #                     = npsr * sum_i log S_i + nfreqs |Gamma_ab|
                return (jnp.block([[jnp.make2d(val * invphi) for val in row] for row in invorf]),
                        phi.shape[0] * orflogdet + orfmat.shape[0] * logdetphi)
                        # was -orfmat.shape[0] * jnp.sum(jnp.log(invphidiag)))
            invprior.params = argmap
            invprior.type = jax.Array

            orfcf = matrix.jsp.linalg.cho_factor(orfmat)
            def factors(params):
                phi = prior(f, df, *[params[arg] for arg in argmap])
                phicf = matrix.jsp.linalg.cho_factor(phi)

                return orfcf, phicf
            factors.params = argmap
        else:
            invprior, factors = None, None
    else:
        def priorfunc(params):
            phis = [prior(f, df, *[params[arg] for arg in argmap]) for prior, argmap in zip(priors, argmaps)]

            return sum(jnp.block([[jnp.make2d(val * phi) for val in row] for row in orfmat])
                       for phi, orfmat in zip(phis, orfmats))
        priorfunc.params = sorted(set.union(*[set(argmap) for argmap in argmaps]))
        priorfunc.type = jax.Array

        invprior, factors = None, None

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmats)
    gp.Phi_inv, gp.factors = invprior, factors

    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})':
                slice(len(f)*i, len(f)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    if means is not None:
        margspec = inspect.getfullargspec(means)
        margs = margspec.args + [arg for arg in margspec.kwonlyargs if arg not in margspec.kwonlydefaults]

        # parameters carried by the pulsar objects (e.g., pos), should be at the beginning of function
        psrpars = [{arg: getattr(psr, arg) for arg in margspec.args if hasattr(psrs[0], arg) and arg not in exclude}
                   for psr in psrs]

        # other means parameters, either common or pulsar-specific
        margmaps = [{arg: f'{meansname}_{arg}' if (f'{meansname}_{arg}' in common or arg in common) else f'{psr.name}_{meansname}_{arg}'
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude} for psr in psrs]

        def meanfunc(params):
            return jnp.concatenate([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp

makegp_fourier_global = makeglobalgp_fourier


# epoch-averaged covariance matrix from covfunc(t1, t2, *args)

def epochavgbasis(psr, components, T=None, dt=1.0):
    bins = quantize(psr.toas, dt)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
    t_avg = psr.toas @ Umat / Umat.sum(axis=0)

    return t_avg, None, Umat

def cov2cov(covfunc):
    argspec = inspect.getfullargspec(covfunc)
    arglist = argspec.args

    if arglist[0] == 't1' and arglist[1] == 't2':
        def covmat(f, df, *args):
            return covfunc(f, f, *args)
    elif arglist[0] == 'tau':
        def covmat(f, df, *args):
            return covfunc(jnp.abs(f[:, jnp.newaxis] - f[jnp.newaxis, :]), *args)
    else:
        raise ValueError('cov2avg() must take a covariance function with arguments t1, t2 or tau.')

    covmat.__signature__ = inspect.signature(covfunc)
    covmat.type = jax.Array

    return covmat

def makegp_avgcov(psr, prior, epochavgbasis=epochavgbasis, common=[], name='avgcovGP'):
    # assume prior(t1, t2, *args) or prior(tau, *args) returns a covariance matrix
    return makegp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                          common=common, exclude=['t1', 't2', 'tau'], name=name)

def makecommongp_avgcov(psrs, prior, epochavgbasis=epochavgbasis, common=[], vector=False, name='avgcovCommonGP'):
    return makecommongp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                                common=common, exclude=['t1', 't2', 'tau'], name=name)

def makeglobalgp_avgcov(psrs, prior, epochavgbasis=epochavgbasis, common=[], vector=False, name='avgcovCommonGP'):
    return makeglobalgp_fourier(psr, cov2cov(prior), components=0, T=1.0, fourierbasis=epochavgbasis,
                                exclude=['t1', 't2', 'tau'], name=name)


# time-interpolated covariance matrix from FFT

def timeinterpbasis(psr, components, T=None, start_time=None):
    if start_time is None:
        start_time = np.min(psr.toas)
    else:
        if start_time > np.min(psr.toas):
            raise ValueError('Coarse time basis start must be earlier than earliest TOA.')

    if T is None:
        T = getspan(psr)

    t_fine = psr.toas
    t_coarse = np.linspace(start_time, start_time + T, components)
    dt_coarse = t_coarse[1] - t_coarse[0]

    idx = np.arange(len(t_fine))
    idy = np.searchsorted(t_coarse, t_fine)
    idy[idy == 0] = 1

    Bmat = np.zeros((len(t_fine), len(t_coarse)), 'd')

    Bmat[idx, idy] = (t_fine - t_coarse[idy - 1]) / dt_coarse
    Bmat[idx, idy - 1] = (t_coarse[idy] - t_fine) / dt_coarse

    return t_coarse, dt_coarse, Bmat

def make_timeinterpbasis(start_time=None, order=1):
    def timeinterpbasis(psr, components, T=None):
        t0 = start_time if start_time is not None else np.min(psr.toas)
        if t0 > np.min(psr.toas):
            raise ValueError('Coarse time basis start must be earlier than earliest TOA.')

        if T is None:
            T = getspan(psr)

        t_fine = psr.toas
        t_coarse = np.linspace(t0, t0 + T, components)
        dt_coarse = t_coarse[1] - t_coarse[0]

        Bmat = si.interp1d(t_coarse, np.identity(components), kind=order)(t_fine).T

        return t_coarse, dt_coarse, Bmat

    return timeinterpbasis

def make_timeinterpbasis_dm(start_time=None, order=1, fref=1400.0):
    """Build a DM time-interpolation basis (fixed chromatic index alpha = 2).

    Time-domain analogue of :func:`make_fourierbasis_dm` used by the FFT-covariance
    GPs: it scales the achromatic :func:`make_timeinterpbasis` basis by the
    cold-plasma dispersion factor ``(fref / psr.freqs) ** 2``. Used by
    :func:`makegp_fftcov_dm`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_dm(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs) ** 2
        return t_coarse, dt_coarse, scale[:, None] * Bmat

    return timeinterpbasis_dm

def make_timeinterpbasis_chromatic(start_time=None, order=1, fref=1400.0):
    """Build a chromatic time-interpolation basis with a variable chromatic index.

    Time-domain analogue of :func:`fourierbasis_chrom` used by the FFT-covariance
    GPs. The returned basis yields a callable ``Bmat_func(alpha)`` that scales the
    achromatic :func:`make_timeinterpbasis` basis by ``(fref / psr.freqs) ** alpha``,
    with ``alpha`` a free parameter. Used by :func:`makegp_fftcov_chrom`.
    """
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_chrom(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs)
        def Bmat_func(alpha):
            return (scale[:, None]**alpha) * Bmat
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_chrom

def make_dmtimeinterpbasis(alpha=2.0, tndm=False, start_time=None, order=1):
    warnings.warn("make_dmtimeinterpbasis is deprecated; use make_timeinterpbasis_dm "
                  "(alpha=2 DM) or make_timeinterpbasis_chromatic (variable alpha) instead.",
                  DeprecationWarning, stacklevel=2)
    basis = make_timeinterpbasis(start_time, order)

    def dmbasis(psr, components, T=None, fref=1400.0):
        t_coarse, dt_coarse, Bmat = basis(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return t_coarse, dt_coarse, Bmat * Dm[:, None]

    return dmbasis

def psd2cov(psdfunc, components, T, oversample=4, fmax_factor=1, cutoff=4):
    if not (isinstance(oversample, int) and isinstance(fmax_factor, int)):
        raise ValueError('psd2cov: oversample and fmax_factor must be integers.')

    if components % 2 == 0:
        raise ValueError('psd2cov: number of components must be odd.')
        
    if cutoff is not None and not isinstance(cutoff, int):
        raise ValueError('psd2cov: cutoff must be None or an integer.')

    if isinstance(cutoff, int) and cutoff > oversample:
        raise ValueError('psd2cov: cutoff must be <= oversample.')

    scaled_components = (components - 1) * fmax_factor + 1
    n_freqs = int((scaled_components - 1) / 2 * oversample + 1)
    fmax = (scaled_components - 1) / T / 2
    freqs = np.linspace(0, fmax, n_freqs)
    df = 1 / T / oversample

    if cutoff is not None:
        i_cutoff = int(np.ceil(oversample / cutoff))
        fs = matrix.jnparray(freqs[i_cutoff:])
    else:
        fs = matrix.jnparray(freqs)

    def covmat(*args):
        clip_max = 1e4
        clip_min = 1e-18
        psd_hi = jnp.clip(psdfunc(fs, 1.0, *args[2:]), clip_min, clip_max)
        
        if cutoff is not None:
            psd = jnp.concatenate([jnp.full((i_cutoff,), psd_hi[0], dtype=psd_hi.dtype), psd_hi])
        else:
            psd = psd_hi

        fullpsd = jnp.concatenate((psd, psd[-2:0:-1]))
        Cfreq = jnp.fft.ifft(fullpsd, norm='backward')
        Ctau = Cfreq.real * len(fullpsd) * df / 2

        return matrix.jsp.linalg.toeplitz(Ctau[:scaled_components:fmax_factor])
    covmat.__signature__ = inspect.signature(psdfunc)
    covmat.type = jax.Array

    return covmat

def makegp_fftcov(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, common=[], name='fftcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T=T,
                          fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                          common=common, name=name)

def makegp_fftcov_dm(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='dm_gp', fref=1400.0):
    """FFT-covariance (time-domain) GP for DM noise (fixed chromatic index alpha = 2).

    DM counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation basis
    is replaced by :func:`make_timeinterpbasis_dm`, scaling each row by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``. ``prior`` is a power-spectral-density
    function (e.g. :func:`powerlaw`) that is converted to a time-domain covariance via
    :func:`psd2cov`. For a free chromatic index use :func:`makegp_fftcov_chrom`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_dm(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_chrom(psr, prior, components, T=None, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, common=[], name='chrom_gp', fref=1400.0):
    """FFT-covariance (time-domain) GP for chromatic noise with a variable index.

    Chromatic counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation
    basis is replaced by :func:`make_timeinterpbasis_chromatic`, scaling each row by
    ``(fref / psr.freqs) ** alpha`` with the chromatic index ``alpha`` a free parameter.
    ``prior`` is a power-spectral-density function (e.g. :func:`powerlaw`) converted to a
    time-domain covariance via :func:`psd2cov`. For the alpha = 2 (DM) case use
    :func:`makegp_fftcov_dm`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_chromatic(start_time=t0, order=order, fref=fref), common=common, name=name)

def makecommongp_fftcov(psrs, prior, components, T, t0=None, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, common=[], vector=False, name='fftcovCommonGP'):
    return makecommongp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T,
                                fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                                common=common, vector=vector, name=name)

def makeglobalgp_fftcov(psrs, prior, orf, components, T, t0, order=1, oversample=4, fmax_factor=1, cutoff=4, fourierbasis=None, name='fftcovGlobalGP'):
    return makeglobalgp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), orf, components, T,
                                fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                                name=name)


# time-interpolated covariance matrix from time-domain

def makegp_intcov(psr, prior, components, T=None, timeinterpbasis=timeinterpbasis, common=[], name='intcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, cov2cov(prior),
                          components, T, fourierbasis=timeinterpbasis, common=common, exclude=['t1', 't2', 'tau'], name=name)

def makecommongp_intcov(psr, prior, components, T, timeinterpbasis=timeinterpbasis, common=[], name='intcovCommonGP'):
    return makecommongp_fourier(psr, cov2cov(prior),
                                components, T, fourierbasis=timeinterpbasis, common=common, exclude=['t1', 't2', 'tau'], name=name)

def makeglobalgp_intcov(psr, prior, orf, components, T, timeinterpbasis=timeinterpbasis, common=[], name='intcovGlobalGP'):
    return makeglobalgp_fourier(psr, cov2cov(prior), orf,
                                components, T, fourierbasis=timeinterpbasis, exclude=['t1', 't2', 'tau'], name=name)


# single powerlaws

def powerlaw(f, df, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

def brokenpowerlaw(f, df, log10_A, gamma, log10_fb):
    kappa = 0.1 # smoothness of transition

    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df * \
        (1.0 + (f / 10.0**log10_fb) ** (1.0 / kappa)) ** (kappa * gamma)

def freespectrum(f, df, log10_rho: typing.Sequence):
    return jnp.repeat(10.0**(2.0 * log10_rho), 2)


def make_combined_crn(components, irn_psd, crn_psd, crn_prefix: typing.Optional[str] = 'crn_'):
    """
    Combine an intrinsic red noise PSD and a common red noise PSD into a
    single PSD function that shares the same Fourier basis.

    The intrinsic red noise PSD is evaluated over the full frequency basis,
    while the common red noise PSD is added only to the first
    ``2 * components`` frequency bins (sine and cosine for each component).

    Parameters
    ----------
    components : int
        Number of shared Fourier frequency components used by the CRN model.
        This determines how many low-frequency bins of the intrinsic basis
        receive the CRN contribution (specifically, the first
        ``2 * components`` entries, corresponding to sine/cosine pairs).
        This is *not* the same as the ``components`` argument passed to
        ``makegp_fourier`` — that controls the total number of Fourier
        components in the basis for the GP (and may be larger, since the
        intrinsic noise can extend to higher frequencies than the CRN).
    irn_psd : callable
        PSD function for the intrinsic red noise. Must accept ``(f, df, ...)``
        and return a PSD array over the full basis.
    crn_psd : callable
        PSD function for the common red noise. Must accept ``(f, df, ...)``
        and return a PSD array. Will only be called on the first
        ``2 * components`` frequency bins.
    crn_prefix : str or None
        Prefix applied to CRN parameter names that overlap with IRN names.
        For example, if both PSDs have ``log10_A`` and ``crn_prefix='crn_'``,
        the combined function will have ``log10_A`` (IRN) and
        ``crn_log10_A`` (CRN) as separate parameters.
        If None, overlapping names are shared (both PSDs receive the same
        value), which is valid when you intentionally want tied parameters.

    Returns
    -------
    combined : callable
        A PSD function whose signature is the union of ``irn_psd`` and
        ``crn_psd`` signatures (with CRN overlaps prefixed). Compatible
        with ``makegp_fourier``: argument names are inspectable via
        ``getfullargspec``, and ``typing.Sequence`` annotations are
        preserved for parameter expansion.
    crn_params : list of str
        The parameter names (as they appear in ``combined``'s signature)
        that belong to the CRN PSD. Pass these directly as the ``common``
        argument to ``makegp_fourier`` or ``makecommongp_fourier`` so that
        the CRN parameters are shared across pulsars rather than given
        per-pulsar names.

        Example::

            combined, crn_params = make_combined_crn(14, ds.powerlaw, ds.powerlaw)
            gp = makegp_fourier(psr, combined, components=30, common=crn_params)
    """
    from discovery import matrix
    irn_spec = inspect.getfullargspec(irn_psd)
    crn_spec = inspect.getfullargspec(crn_psd)

    shared = {'f', 'df'}
    irn_names = [a for a in irn_spec.args if a not in shared]
    crn_names = [a for a in crn_spec.args if a not in shared]

    # Rename overlapping CRN params
    irn_set = set(irn_names)
    crn_rename = {}  # original_name -> merged_name
    for a in crn_names:
        if a in irn_set and crn_prefix is not None:
            crn_rename[a] = crn_prefix + a
        else:
            crn_rename[a] = a

    # Build merged argument list: f, df, irn params, then (renamed) crn params
    merged_args = ['f', 'df']
    seen = set(shared)
    for arg in irn_names:
        if arg not in seen:
            merged_args.append(arg)
            seen.add(arg)
    for arg in crn_names:
        renamed = crn_rename[arg]
        if renamed not in seen:
            merged_args.append(renamed)
            seen.add(renamed)

    # Merge annotations (applying rename to CRN annotations)
    annotations = {}
    if irn_spec.annotations:
        annotations.update({k: v for k, v in irn_spec.annotations.items()
                            if k not in shared})
    if crn_spec.annotations:
        for k, v in crn_spec.annotations.items():
            if k not in shared:
                annotations[crn_rename.get(k, k)] = v

    def _impl(f, df, kw):
        irn_kw = {k: kw[k] for k in irn_names}
        crn_kw = {k: kw[crn_rename[k]] for k in crn_names}
        if matrix.jnp == jnp:
            phi = irn_psd(f, df, **irn_kw)
            phi = phi.at[:2 * components].add(
                crn_psd(f[:2 * components], df[:2 * components], **crn_kw)
            )
        else:
            phi = irn_psd(f, df, **irn_kw)
            phi[:2 * components] += crn_psd(
                f[:2 * components], df[:2 * components], **crn_kw
            )
        return phi

    # Dynamically build a function with the correct inspectable signature
    param_args = merged_args[2:]
    args_str = ', '.join(merged_args)
    kwargs_dict = '{' + ', '.join(f"'{a}': {a}" for a in param_args) + '}'
    func_code = f"def combined({args_str}): return _impl(f, df, {kwargs_dict})"
    ns = {'_impl': _impl}
    exec(func_code, ns)
    combined = ns['combined']
    combined.__annotations__ = annotations

    # Deduplicated list of CRN param names as they appear in the combined signature
    crn_params = list(dict.fromkeys(crn_rename[k] for k in crn_names))

    return combined, crn_params



# combined red_noise + crn

# this is a factory because it needs to specify a different number of components for the CRN
# note that the preferred way to fix gamma is for the user to use matrix.partial directly
def makepowerlaw_crn(components, crn_gamma='variable'):
    if matrix.jnp == jnp:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi = phi.at[:2*components].add((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                            const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi
    elif matrix.jnp == np:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / np.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi[:2*components] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / np.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi

    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn

def powerlaw_brokencrn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb):
    kappa = 0.1 # smoothness of transition

    phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
    return phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) * df * \
        (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)

def brokenpowerlaw_brokencrn(f, df, log10_A, gamma, log10_fb, crn_log10_A, crn_gamma, crn_log10_fb):
    kappa = 0.1 # smoothness of transition

    phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df * \
        (1 + (f / 10**log10_fb) ** (1 / kappa)) ** (kappa * gamma)
    return phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) * df * \
        (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)

def makefreespectrum_crn(components):
    if matrix.jnp == jnp:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = jnp.repeat(10.0**(2.0 * log10_rho), 2)
            phi = phi.at[:2*components].add(jnp.repeat(10.0**(2.0 * crn_log10_rho), 2))
            return phi
    elif matrix.jnp == np:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = jnp.repeat(10.0**(2.0 * log10_rho), 2)
            phi[:2*components] += jnp.repeat(10.0**(2.0 * crn_log10_rho), 2)
            return phi

    return freespectrum_crn


# ORFs: OK as numpy functions

def uncorrelated_orf(pos1, pos2):
    return 1.0 if np.all(pos1 == pos2) else 0.0

def hd_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0
    else:
        omc2 = (1.0 - np.dot(pos1, pos2)) / 2.0
        return 1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5

def monopole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        # conditioning trick from enterprise
        return 1.0 + 1.0e-6
    else:
        return 1.0

def dipole_orf(pos1, pos2):
    if np.all(pos1 == pos2):
        return 1.0 + 1.0e-6
    else:
        return np.dot(pos1, pos2)


def makedelay(psr, delay, components=None, common=[], name='delay'):
    argspec = inspect.getfullargspec(delay)
    args = argspec.args + [arg for arg in argspec.kwonlyargs if arg not in argspec.kwonlydefaults]

    argmap = {arg: (arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({components})' if (argspec.annotations.get(arg) == typing.Sequence and components is not None) else '')
              for arg in args if not hasattr(psr, arg)}

    psrpars = {arg: matrix.jnparray(getattr(psr, arg)) for arg in args if hasattr(psr, arg)}

    def delayfunc(params):
        return delay(**psrpars, **{arg: params[argname] for arg,argname in argmap.items()})
    delayfunc.params = sorted(argmap.values())

    return delayfunc

# use with makedelay to set residuals dynamically from arrays
def getresiduals(y):
    return -y
