from functools import partial
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


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, ecorr=False, chromequad=False,
                          chromequad_idx_per_backend=False,
                          selection=selection_backend_flags, vectorize=True,
                          outliers=False, enterprise=False, fref=1400):
    """Build a measurement noise matrix for a pulsar.

    The noise variance per TOA is (tnequad=True):
        efac^2 * (scale * toaerr)^2 + EQUAD^2 [+ CHROMEQUAD^2 * (fref/freq)^chrom_idx]

    or (tnequad=False, t2equad convention):
        efac^2 * ((scale * toaerr)^2 + EQUAD^2) [+ CHROMEQUAD^2 * (fref/freq)^chrom_idx]

    Parameters
    ----------
    chromequad : bool
        If True, add a per-backend frequency-dependent noise floor term.
    chromequad_idx_per_backend : bool
        If True, float a separate chrom_idx for each backend.
        Default False uses a single per-pulsar chrom_idx (better constrained).
    fref : float
        Reference frequency in MHz for the chromatic scaling.
    """
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    if tnequad:
        log10_equads = [f'{psr.name}_{backend}_log10_tnequad' for backend in backends]
    else:
        log10_equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]

    params = efacs + log10_equads

    if chromequad:
        log10_chromequads = [f'{psr.name}_{backend}_log10_chromequad' for backend in backends]
        params = params + log10_chromequads
        if chromequad_idx_per_backend:
            chromequad_idxs = [f'{psr.name}_{backend}_chromequad_idx' for backend in backends]
        else:
            chromequad_idxs = [f'{psr.name}_chromequad_idx']
        params = params + chromequad_idxs

    masks = [(backend_flags == backend) for backend in backends]
    logscale = np.log10(scale)

    if outliers:
        toaerr_scaling = f'{psr.name}_alpha_scaling({psr.toas.size})'
        params.append(toaerr_scaling)

    def _chrom_scaling_fixed(backend_idx):
        """Return per-TOA chromatic scaling array for a backend using noisedict values."""
        if chromequad_idx_per_backend:
            idx = noisedict[chromequad_idxs[backend_idx]]
        else:
            idx = noisedict[chromequad_idxs[0]]
        cq2 = 10.0**(2 * noisedict[log10_chromequads[backend_idx]])
        return cq2 * (fref / psr.freqs)**idx

    if all(par in noisedict for par in params):
        if outliers:
            raise ValueError("No outlier scaling if white noise is fixed.")
        if tnequad:
            if chromequad:
                noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2
                                    + 10.0**(2 * (logscale + noisedict[log10_equad]))
                                    + _chrom_scaling_fixed(i))
                            for i, (mask, efac, log10_equad) in enumerate(zip(masks, efacs, log10_equads)))
            else:
                noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2
                                    + 10.0**(2 * (logscale + noisedict[log10_equad])))
                            for mask, efac, log10_equad in zip(masks, efacs, log10_equads))
        else:
            if chromequad:
                noise = sum(mask * (noisedict[efac]**2 * ((scale * psr.toaerrs)**2
                                                           + 10.0**(2 * (logscale + noisedict[log10_equad])))
                                    + _chrom_scaling_fixed(i))
                            for i, (mask, efac, log10_equad) in enumerate(zip(masks, efacs, log10_equads)))
            else:
                noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2
                                                          + 10.0**(2 * (logscale + noisedict[log10_equad])))
                            for mask, efac, log10_equad in zip(masks, efacs, log10_equads))

        if ecorr:
            egp = makegp_ecorr(psr, noisedict=noisedict, enterprise=enterprise, scale=scale, selection=selection)
            return matrix.NoiseMatrixSM_novar(noise, egp.F, egp.Phi.N)
        else:
            return matrix.NoiseMatrix1D_novar(noise)
    else:
        if vectorize:
            toaerrs2 = matrix.jnparray(scale**2 * psr.toaerrs**2)
            masks_jnp = matrix.jnparray([mask for mask in masks])

            if chromequad:
                freqs_jnp = matrix.jnparray(psr.freqs)

            if tnequad:
                def getnoise(params):
                    alpha_scaling = params[toaerr_scaling] if outliers else 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_equad]))
                                              for log10_equad in log10_equads])
                    base = (masks_jnp * (efac2[:, jnp.newaxis] * (alpha_scaling * toaerrs2)[jnp.newaxis, :]
                                         + equad2[:, jnp.newaxis])).sum(axis=0)
                    if chromequad:
                        if chromequad_idx_per_backend:
                            idxs = jnp.array([params[ci] for ci in chromequad_idxs])
                        else:
                            idxs = jnp.full(len(backends), params[chromequad_idxs[0]])
                        cq2 = jnp.array([10.0**(2 * params[lc]) for lc in log10_chromequads])
                        freq_scale = (fref / freqs_jnp[jnp.newaxis, :])**idxs[:, jnp.newaxis]
                        base = base + (masks_jnp * cq2[:, jnp.newaxis] * freq_scale).sum(axis=0)
                    return base
            else:
                def getnoise(params):
                    alpha_scaling = params[toaerr_scaling] if outliers else 1.0
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_equad]))
                                              for log10_equad in log10_equads])
                    base = (masks_jnp * efac2[:, jnp.newaxis]
                            * ((alpha_scaling * toaerrs2)[jnp.newaxis, :] + equad2[:, jnp.newaxis])).sum(axis=0)
                    if chromequad:
                        if chromequad_idx_per_backend:
                            idxs = jnp.array([params[ci] for ci in chromequad_idxs])
                        else:
                            idxs = jnp.full(len(backends), params[chromequad_idxs[0]])
                        cq2 = jnp.array([10.0**(2 * params[lc]) for lc in log10_chromequads])
                        freq_scale = (fref / freqs_jnp[jnp.newaxis, :])**idxs[:, jnp.newaxis]
                        base = base + (masks_jnp * cq2[:, jnp.newaxis] * freq_scale).sum(axis=0)
                    return base
        else:
            toaerrs = matrix.jnparray(scale * psr.toaerrs)
            masks_list = [matrix.jnparray(mask) for mask in masks]

            if chromequad:
                freqs_jnp = matrix.jnparray(psr.freqs)

            if tnequad:
                def getnoise(params):
                    alpha_scaling = params[toaerr_scaling] if outliers else 1.0
                    base = sum(mask * (params[efac]**2 * (alpha_scaling * toaerrs)**2
                                       + 10.0**(2 * (logscale + params[log10_equad])))
                               for mask, efac, log10_equad in zip(masks_list, efacs, log10_equads))
                    if chromequad:
                        for i, (mask, lc) in enumerate(zip(masks_list, log10_chromequads)):
                            ci = chromequad_idxs[i] if chromequad_idx_per_backend else chromequad_idxs[0]
                            base = base + mask * 10.0**(2 * params[lc]) * (fref / freqs_jnp)**params[ci]
                    return base
            else:
                def getnoise(params):
                    alpha_scaling = params[toaerr_scaling] if outliers else 1.0
                    base = sum(mask * params[efac]**2 * (alpha_scaling * toaerrs**2
                                                          + 10.0**(2 * (logscale + params[log10_equad])))
                               for mask, efac, log10_equad in zip(masks_list, efacs, log10_equads))
                    if chromequad:
                        for i, (mask, lc) in enumerate(zip(masks_list, log10_chromequads)):
                            ci = chromequad_idxs[i] if chromequad_idx_per_backend else chromequad_idxs[0]
                            base = base + mask * 10.0**(2 * params[lc]) * (fref / freqs_jnp)**params[ci]
                    return base

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
    log10_ecorrs, Umats = [], []

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')


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
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))
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

# chromatic quadratic filter closed over a variable chromatic index
def chromatic_quad_basis(
        psr,
        fref: float = 1400.0,
        chrom_idx: typing.Optional[float] = None,
) -> typing.Union[jnp.ndarray, typing.Callable]:
    """Normalized quadratic basis for a chromatic (radio-frequency dependent) process.

    Analogous to the DM, DM1, DM2 timing-model terms but for an arbitrary chromatic
    index. Adding it as a GP rather than through the timing model lets the chromatic
    index parameter be shared with a free chromatic Fourier GP (see
    :func:`fourierbasis_chrom`) by giving both signals the same ``name``.


    Parameters
    ----------
    psr : :class:`pulsar.Pulsar`
        Discovery Pulsar object containing TOAs and radio frequencies.
    fref : float, optional
        Reference frequency in MHz for the chromatic scaling. Default is 1400.0.
    chrom_idx : float, optional
        Chromatic (radio-frequency) index. If ``None`` (default) a callable
        ``fmat(chrom_idx) -> jnp.ndarray`` is returned so the index can be a free
        parameter; if a value is given, the fixed :math:`N_\\mathrm{TOA} \\times 3`
        basis matrix is returned directly.

    Returns
    -------
    Callable or jnp.ndarray
        When *chrom_idx* is ``None``, a function ``fmat(chrom_idx)`` returning the
        normalized :math:`N_\\mathrm{TOA} \\times 3` basis (with a ``.ncol`` attribute
        giving the column count). Otherwise the evaluated basis matrix.
    """
    ret = np.zeros((len(psr.toas), 3))
    t0 = (psr.toas.max() + psr.toas.min()) / 2
    for ii in range(3):
        ret[:, ii] = (psr.toas - t0) ** (ii)
    f_scale = (fref / psr.freqs)

    def fmat_func(chrom_idx):
        retp = ret * f_scale[:, None] ** chrom_idx
        norm = jnp.sqrt(jnp.sum(retp**2, axis=0))
        return retp / norm
    fmat_func.ncol = ret.shape[1]

    if chrom_idx is not None:
        return fmat_func(chrom_idx)
    return fmat_func

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
    if isinstance(components, (int, np.integer)):
        f  = np.arange(1, components + 1, dtype=np.float64) / T
    elif isinstance(components, (list, tuple, np.ndarray)):
        f = np.array(components, dtype=np.float64)
    else:
        raise TypeError(f"`components` must be an int (number of Fourier modes) or an "
                        f"array-like of explicit modes, got {type(components).__name__}.")
    components = len(f)
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*components), dtype=np.float64)
    for i in range(components):
        fmat[:, 2*i  ] = np.sin(2.0 * np.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * np.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat


def log_fourierbasis(psr, T=None, logmode=-1, f_min=None, nlin=30, nlog=0):
    if T is None:
        T = getspan(psr)

    f, w_lin = linBinning(T, logmode, f_min, nlin, nlog)

    #f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*len(f)), dtype=np.float64)
    for i in range(len(f)):
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def log_fourierbasis_dm(psr, T=None, logmode=-1, f_min=None, nlin=30, nlog=0, fref=1400):
    if T is None:
        T = getspan(psr)

    f, w_lin = linBinning(T, logmode, f_min, nlin, nlog)

    #f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*len(f)), dtype=np.float64)
    for i in range(len(f)):
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    Dm = (fref / psr.freqs)**2

    return np.repeat(f, 2), np.repeat(df, 2), fmat * Dm[:, None]

def log_fourierbasis_chrom(psr, T=None, logmode=-1, f_min=None, nlin=30, nlog=0, fref=800):
    if T is None:
        T = getspan(psr)

    f, w_lin = linBinning(T, logmode, f_min, nlin, nlog)

    #f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*len(f)), dtype=np.float64)
    for i in range(len(f)):
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(alpha):
        return fmat * fnorm[:, None]**alpha

    return np.repeat(f, 2), np.repeat(df, 2), fmatfunc

def log_fourierbasis_chrom_fixed(psr, alpha = 4.0, T=None, logmode=-1, f_min=None, nlin=30, nlog=0, fref=800):
    if T is None:
        T = getspan(psr)

    f, w_lin = linBinning(T, logmode, f_min, nlin, nlog)

    #f  = np.arange(1, components + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2*len(f)), dtype=np.float64)
    for i in range(len(f)):
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    fmat = fmat * fnorm[:, None]**alpha

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def linBinning(T, logmode, f_min, nlin, nlog):
    """
    Copied from enterprise_extensions.
    Get the frequency binning for the low-rank approximations, including
    log-spaced low-frequency coverage.
    Credit: van Haasteren & Vallisneri, MNRAS, Vol. 446, Iss. 2 (2015)

    :param T:       Duration experiment
    :param logmode: From which linear mode to switch to log
    :param f_min:   Down to which frequency we'll sample
    :param nlin:    How many linear frequencies we'll use
    :param nlog:    How many log frequencies we'll use

    """
    if logmode < 0:
        raise ValueError(
            "Cannot do log-spacing when all frequencies are" "linearly sampled"
        )

    # First the linear spacing and weights
    df_lin = 1.0 / T
    f_min_lin = (1.0 + logmode) / T
    f_lin = jnp.linspace(f_min_lin, f_min_lin + (nlin - 1) * df_lin, nlin)
    w_lin = jnp.sqrt(df_lin * jnp.ones(nlin))

    if nlog > 0:
        # Now the log-spacing, and weights
        f_min_log = jnp.log(f_min)
        f_max_log = jnp.log((logmode + 0.5) / T)
        df_log = (f_max_log - f_min_log) / (nlog)
        f_log = jnp.exp(
            jnp.linspace(f_min_log + 0.5 * df_log, f_max_log - 0.5 * df_log, nlog)
        )
        w_log = jnp.sqrt(df_log * f_log)
        return jnp.append(f_log, f_lin), jnp.append(w_log, w_lin)
    else:
        return f_lin, w_lin

# Time domain kernels (covariances)

def ridge_kernel(
        log10_sigma_ridge: float = -7.,
) -> typing.Callable:
    """Ridge (diagonal) covariance kernel factory.

    Parameters
    ----------
    log10_sigma_ridge : float
        Log10 of the amplitude; diagonal entries are
        :math:`\\sigma^2 = 10^{2\\,\\texttt{log10\_sigma\_ridge}}`.

    Returns
    -------
    Callable
        A function ``kernel(tau) -> jnp.ndarray`` returning the
        :math:`N \\times N` diagonal covariance matrix for an *N*-element
        lag vector *tau*.
    """
    def kernel(tau, log10_sigma_ridge=log10_sigma_ridge):
        scale = 10**(2 * log10_sigma_ridge)
        return scale * jnp.eye(len(tau), dtype=tau.dtype)

    return kernel

def square_exponential_kernel(
        log10_sigma_sq_exp: float = -7.,
        log10_ell: float = 1.,
) -> typing.Callable:
    """Squared-exponential (RBF) covariance kernel factory.

    Parameters
    ----------
    log10_sigma_sq_exp : float
        Log10 of the amplitude.
    log10_ell : float
        Log10 of the length scale in **days**.

    Returns
    -------
    Callable
        A function ``kernel(tau) -> jnp.ndarray`` returning the
        :math:`N \\times N` covariance matrix for an *N*-element lag vector
        *tau* in seconds.

    Notes
    -----
    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(-\\frac{\\tau^2}{2\\ell^2}\\right)
            + d\\,\\delta_{ij}

    where :math:`d = (\\sigma / 50000)^2` is a small diagonal regulariser.
    """
    def kernel(tau, log10_sigma_sq_exp=log10_sigma_sq_exp, log10_ell=log10_ell):
        sigma2 = 10**(2 * log10_sigma_sq_exp)
        ell = 10**log10_ell * 86400  # days -> seconds
        sigma = 10**log10_sigma_sq_exp
        d = jnp.eye(len(tau), dtype=tau.dtype) * (sigma / 50000.)**2
        return sigma2 * jnp.exp(-0.5 * (tau / ell)**2) + d

    return kernel

def quasi_periodic_kernel(
        log10_sigma_quasi_periodic: float = -7.,
        log10_ell: float = 1.,
        log10_gamma_p: float = 0.,
        log10_p: float = 0.,
) -> typing.Callable:
    """Quasi-periodic (SE × periodic) covariance kernel factory.

    Matches the ``periodic_kernel`` convention in enterprise_extensions.

    Parameters
    ----------
    log10_sigma_quasi_periodic : float
        Log10 of the amplitude.
    log10_ell : float
        Log10 of the squared-exponential length scale in **days**.
    log10_gamma_p : float
        Log10 of the periodic damping amplitude (direct scale: larger →
        stronger periodic decay, matching enterprise convention).
    log10_p : float
        Log10 of the period in **years**.

    Returns
    -------
    Callable
        A function ``kernel(tau) -> jnp.ndarray`` returning the
        :math:`N \\times N` covariance matrix for an *N*-element lag vector
        *tau* in seconds.

    Notes
    -----
    .. math::

        K(\\tau) = \\sigma^2 \\exp\\!\\left(
            -\\frac{\\tau^2}{2\\ell^2}
            - \\gamma_p \\sin^2\\!\\left(\\frac{\\pi\\tau}{p}\\right)
        \\right) + d\\,\\delta_{ij}

    where :math:`d = (\\sigma / 50000)^2` is a small diagonal regulariser.
    """
    def kernel(tau, log10_sigma_quasi_periodic=log10_sigma_quasi_periodic, log10_ell=log10_ell,
               log10_gamma_p=log10_gamma_p, log10_p=log10_p):
        sigma2 = 10**(2 * log10_sigma_quasi_periodic)
        ell = 10**log10_ell * 86400  # days -> seconds
        gamma_p = 10**log10_gamma_p
        p = 10**log10_p * 365.25 * 86400  # years -> seconds
        sigma = 10**log10_sigma_quasi_periodic
        d = jnp.eye(len(tau), dtype=tau.dtype) * (sigma / 50000.)**2
        return sigma2 * jnp.exp(-0.5 * (tau / ell)**2 - gamma_p * jnp.sin(jnp.pi * tau / p)**2) + d

    return kernel


def matern_kernel(
        log10_sigma_matern: float = -7.,
        log10_ell: float = 1.,
        nu: float = 1.5,
) -> typing.Callable:
    """Matérn covariance kernel factory.

    Parameters
    ----------
    log10_sigma_matern : float
        Log10 of the amplitude.
    log10_ell : float
        Log10 of the length scale in **days**.
    nu : float
        Smoothness parameter; must be one of ``{0.5, 1.5, 2.5}``.

    Returns
    -------
    Callable
        A function ``kernel(tau) -> jnp.ndarray`` returning the
        :math:`N \\times N` covariance matrix for an *N*-element lag vector
        *tau* in seconds.

    Raises
    ------
    ValueError
        If *nu* is not in ``{0.5, 1.5, 2.5}``.

    Notes
    -----
    Supports the Matérn-½ (``nu=0.5``), Matérn-3/2 (``nu=1.5``), and
    Matérn-5/2 (``nu=2.5``) closed-form kernels.  A small diagonal
    regulariser :math:`d = (\\sigma / 50000)^2` is added for numerical
    stability.
    """

    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("matern_kernel currently supports nu in {0.5, 1.5, 2.5}.")

    def kernel(tau, log10_sigma_matern=log10_sigma_matern, log10_ell=log10_ell,):
        sigma2 = 10**(2 * log10_sigma_matern)
        ell = 10**log10_ell * 86400  # days -> seconds
        r = jnp.abs(tau) / ell

        if nu == 0.5:
            k = jnp.exp(-r)
        elif nu == 1.5:
            c = jnp.sqrt(3.0)
            k = (1.0 + c * r) * jnp.exp(-c * r)
        else:  # nu == 2.5
            c = jnp.sqrt(5.0)
            k = (1.0 + c * r + (5.0 / 3.0) * r**2) * jnp.exp(-c * r)

        sigma = 10**log10_sigma_matern
        d = jnp.eye(len(tau), dtype=tau.dtype) * (sigma / 50000.)**2
        return sigma2 * k + d

    return kernel

# time domain interpolation bases

def linear_blocked_interpolation_basis(
        toas,
        bin_edges,
):
    bin_edges = bin_edges * 86400 # MJD to seconds
    # uses a custom set of bin_edges
    M = np.zeros((len(toas), len(bin_edges)))
    # make linear interpolation basis
    for ii in range(len(bin_edges) - 1):
        idx = np.logical_and(toas >= bin_edges[ii], toas <= bin_edges[ii + 1])
        M[idx, ii] = (toas[idx] - bin_edges[ii + 1]) / (bin_edges[ii] - bin_edges[ii + 1])
        M[idx, ii + 1] = (toas[idx] - bin_edges[ii]) / (bin_edges[ii + 1] - bin_edges[ii])

    # only return non-zero columns for rank reduction
    idx = M.sum(axis=0) != 0

    return M[:, idx], bin_edges[idx]


def custom_blocked_interpolation_basis(
        toas,
        nodes,
        kind="linear",
):
    nodes = nodes * 86400  # MJD to seconds
    basis = np.identity(len(nodes))
    interp = si.interpolate.interp1d(
        nodes,
        basis,
        kind=kind,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    M = interp(toas)
    # only return non-zero columns for rank reduction
    idx = M.sum(axis=0) != 0
    if not np.any(idx):
        raise RuntimeError(
            "Interpolation basis has no support in the TOA range. Perhaps check units."
        )

    return M[:, idx], nodes[idx]

def makegp_improper_varF(psr, fmat, constant=1.0e40, name='improperGP_varF',
                         param_names=[], noisedict={}, project=None):
    """Improper GP with a parameter-dependent design matrix.
    Like :func:`makegp_improper`, but the design matrix comes from a callable basis
    whose columns depend on fit parameters -- for example :func:`chrom_poly_basis`,
    whose columns depend on the chromatic index. The varying parameter is named
    ``{psr.name}_{name}_{param}``, so it is shared with any other signal carrying the
    same name, such as a chromatic Fourier GP.
    The timing-model column span is removed and the basis is orthonormalised at every
    evaluation. Neither is optional: the timing model carries an improper prior over its
    own directions, and an improper prior over a basis whose scale varies with the
    parameters scores them on that scale rather than on the data.
    psr:            Discovery Pulsar object
    fmat:           basis factory ``fmat(*param_values) -> (N_toa, N_col)`` array. May
                    carry an ``ncol`` attribute giving its column count; if absent the
                    width is found by evaluating it once
    constant:       diagonal of the flat improper prior over the coefficients
    name:           base name for the GP parameters
    param_names:    names of the parameters passed positionally to fmat
    noisedict:      fixed parameter values; if every entry of param_names is present
                    the basis is evaluated once and a ConstantGP returned, otherwise a
                    VariableGP whose design matrix varies with the free parameters
    project:        further bases to remove alongside the timing model, each an array
                    or a GP with a non-callable ``F``
    """
    # factorised once: the timing model does not depend on the fit parameters
    Q_null, _ = np.linalg.qr(normalise_tm_basis(psr))

    if project is not None:
        parts = project if isinstance(project, (list, tuple)) else [project]
        mats = []
        for p in parts:
            F_p = getattr(p, 'F', p)
            if callable(F_p):
                raise ValueError(
                    f'makegp_improper_varF: {psr.name}: a basis passed to project has a '
                    f'callable F, so it has no fixed column span to remove. Only bases '
                    f'with a constant design matrix can be projected out.')
            mats.append(np.asarray(F_p, dtype=np.float64))
        P = np.hstack(mats)
        P = P - Q_null @ (Q_null.T @ P)
        Up, Sp, _ = np.linalg.svd(P, full_matrices=False)
        Q_null = np.hstack([Q_null, Up[:, Sp > 1e-10 * Sp[0]]])

    Q_j = matrix.jnparray(Q_null)

    def shape_np(F):
        return np.linalg.qr(F - Q_null @ (Q_null.T @ F))[0]

    def shape_jnp(F):
        return jnp.linalg.qr(F - Q_j @ (Q_j.T @ F))[0]

    ncol = getattr(fmat, 'ncol', None)
    if ncol is None:
        ncol = np.asarray(fmat(*[1.0 for _ in param_names])).shape[1]

    # noisedict keys are the full parameter names, as everywhere else in this module,
    # so that a dict taken straight from a single-pulsar chain fixes the basis
    argmap = [f'{psr.name}_{name}_{param}' for param in param_names]

    if all(arg in noisedict for arg in argmap):
        F_const = shape_np(np.asarray(fmat(*[noisedict[arg] for arg in argmap]),
                                      dtype=np.float64))
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(ncol)),
                               F_const)
    else:
        phi = matrix.jnparray(constant * np.ones(ncol))

        def getphi(params):
            return phi
        getphi.params = []

        def get_fmat(params):
            return shape_jnp(fmat(*[params[arg] for arg in argmap]))
        get_fmat.params = argmap

        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), get_fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({ncol})': slice(0, ncol)}

    gp.name, gp.pos, gp.gpname, gp.gpcommon = psr.name, psr.pos, name, []

    return gp

def normalise_tm_basis(psr, scale=1.0):
    """Timing-model design matrix with unit-norm columns.
    All-zero columns, which arise when a fitted par-file parameter has no TOAs
    behind it, are dropped and reported. Dividing by their zero norm would give
    NaNs, and they span nothing, so removing them leaves the column space
    unchanged.
    """
    Mmat = np.asarray(scale * psr.Mmat, dtype=np.float64)
    norms = np.sqrt(np.sum(Mmat**2, axis=0))
    keep = norms > 0

    ndrop = int((~keep).sum())
    if ndrop:
        idx = np.where(~keep)[0]
        names = (list(np.asarray(psr.fitpars)[idx]) if hasattr(psr, 'fitpars')
                 else list(idx))
        print(f'Warning: {psr.name} has {ndrop} all-zero timing-model column(s), '
              f'dropped: {names}')

    return Mmat[:, keep] / norms[keep]

def chrom_poly_basis(psr, fref=None):
    """Callable chromatic polynomial basis ``U * (fref/freq)**alpha``.
    ``U`` is the SVD-orthonormalised [1, t, t**2] temporal design matrix. The SVD is a
    fixed right-multiplication of the raw polynomial, so it leaves the column span, and
    hence the marginal likelihood under an orthonormalising GP, unchanged.
    Returns ``fmat(alpha) -> (N_toa, 3)``, carrying ``ncol``, the reference frequency
    ``fref`` and the temporal ``svd`` factors, for use with
    :func:`makegp_improper_varF`.
    psr:  Discovery Pulsar object
    fref: reference frequency; defaults to the geometric mean of the TOA frequencies
    """
    t0_sec  = float(np.mean(psr.toas))
    toas_yr = (psr.toas - t0_sec) / const.yr

    if fref is None:
        # Geometric mean of observing frequencies
        fref = float(np.exp(np.mean(np.log(np.asarray(psr.freqs)))))

    M_poly = np.vstack([np.ones_like(toas_yr), toas_yr, toas_yr**2]).T
    U, S, Vt = np.linalg.svd(M_poly, full_matrices=False)

    U_j     = matrix.jnparray(U)
    fnorm_j = matrix.jnparray(fref / np.asarray(psr.freqs))

    def fmat(alpha):
        return U_j * fnorm_j[:, None] ** alpha
    fmat.ncol = 3
    fmat.fref = fref
    fmat.svd = {'S': S, 'Vt': Vt}

    return fmat

def makegp_chrom_poly_svd(psr, fref=None, constant=1e40, name='chrom_gp', project=None,
                          noisedict={}):
    """SVD-orthogonalised chromatic polynomial GP, marginalised analytically.
    A :func:`chrom_poly_basis` carried by :func:`makegp_improper_varF`, so the timing
    model is projected out and the basis orthonormalised at every alpha.
    Shares ``alpha`` with a companion chromatic Fourier (or FFTint) GP via the
    parameter name ``{psr}_{name}_alpha``.
    psr:       Discovery Pulsar object
    fref:      reference frequency; defaults to the geometric mean of the TOA frequencies
    constant:  diagonal of the flat improper prior over the coefficients
    name:      base name for the GP parameters
    project:   further bases to remove alongside the timing model -- an array or a GP
               with a non-callable ``F``
    noisedict: fixed value for ``{psr}_{name}_alpha``; if present the basis is
               evaluated once and a ConstantGP returned
    """
    fmat = chrom_poly_basis(psr, fref=fref)

    gp = makegp_improper_varF(psr, fmat, constant=constant, name=name,
                              param_names=['alpha'], noisedict=noisedict,
                              project=project)
    gp.svd = fmat.svd

    return gp

def makegp_timedomain_dm(psr, covariance, dt=1.0, Umat=None, nodes=None, common=[], name='dm_gp', fref=1400, noisedict={}):
    """
    Construct a time-domain Gaussian process for dispersion measure variations.

    This function builds a GP model for DM variations by combining
    a covariance function in the time domain with a model for the DM variations.
    The TOAs are quantized into time bins, and the GP is constructed using the time separations
    between bins weighted by the DM signature.

    Parameters
    ----------
    psr : :class:`pulsar.Pulsar`
        Discovery Pulsar object containing TOAs and radio frequencies.
    covariance : callable
        Function that returns the time domain autocorrelation for a given
        separation (tau). Must have signature `covariance(tau, *params)` where
        tau is the time separation array.
    dt : float, optional
        Time bin width in seconds for quantizing TOAs. Default is 1.0.
    Umat : ndarray, optional
        Design matrix mapping the low-rank GP to the TOA residuals. If None,
        it will be constructed by quantizing the TOAs and weighting by the DM signature.
        Default is None.
    common : list, optional
        List of parameter names that should be treated as common (shared) across
        pulsars rather than pulsar-specific. Default is [].
    name : str, optional
        Base name for the GP parameters. Used as prefix for parameter naming.
        Default is 'dm_gp'.
    fref : float, optional
        Reference frequency in MHz for scaling the DM signature. Default is 1400 MHz.
    noisedict : dict, optional
        If it supplies values for all of the GP's hyperparameters (fixed-point
        analysis), the covariance is evaluated once and a :class:`matrix.ConstantGP`
        is returned so its contribution to the covariance is cached rather than
        sampled; otherwise a :class:`matrix.VariableGP` is returned. Default is {}.

    Returns
    -------
    :class:`matrix.VariableGP` or :class:`matrix.ConstantGP`
        A GP object containing the noise covariance matrix (as a NoiseMatrix2D_var,
        or NoiseMatrix2D_novar when the hyperparameters are fixed via noisedict) and
        the design matrix (Umat) that maps the GP to the TOA residuals via DM delays.
        See :class:`matrix.VariableGP` / :class:`matrix.ConstantGP`.

    Notes
    -----
    The design matrix Umat maps the low-rank GP (evaluated at quantized TOAs)
    to the full TOA residuals, scaled by the frequency-dependent DM signature.
    """
    # Lazy import to avoid circular dependency
    from discovery.signals import quantize

    argspec = inspect.getfullargspec(covariance)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args if arg not in ['tau']]

    # get radio frequency scaling
    dt_DM = (fref / psr.freqs)**(2.0)

    if Umat is None:
        bins = quantize(psr.toas, dt)
        Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
        Umat = Umat * dt_DM[:, None]
        nodes = psr.toas @ Umat / Umat.sum(axis=0)
    else:
        Umat = Umat * dt_DM[:, None]
        assert nodes is not None, "If Umat is provided, nodes must also be provided."

    get_tmat = covariance
    tau = jnp.abs(nodes[:, jnp.newaxis] - nodes[jnp.newaxis, :])

    def getphi(params):
        return get_tmat(tau, *[params[arg] for arg in argmap])
    getphi.params = argmap

    # fixed noise parameter analysis: if all hyperparameters are supplied in noisedict,
    # evaluate the covariance once and return a cached ConstantGP.
    if noisedict and all(par in noisedict for par in argmap):
        gp = matrix.ConstantGP(matrix.NoiseMatrix2D_novar(getphi(noisedict)), Umat)
    else:
        gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)
    gp.index = {f'{psr.name}_{name}_coefficients({Umat.shape[1]})': slice(0, Umat.shape[1])}
    return gp

def fourierbasis_dm(psr, components, T=None, fref=1400.0):
    """Fourier design matrix for a DM (dispersion measure) Gaussian process.

    Identical to :func:`fourierbasis`, but each row is scaled by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``, i.e. a fixed chromatic index
    alpha = 2. Use :func:`fourierbasis_chrom` when the chromatic index is a free
    parameter, in which case the process is general chromatic noise rather than DM.
    """
    f, df, fmat = fourierbasis(psr, components, T=T)

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
    f, df, fmat = fourierbasis(psr, components, T=T)

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
        f, df, fmat = fourierbasis(psr, components, T=T)

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

def make_dmfourierbasis(alpha=2.0, tndm=False):
    warnings.warn("make_dmfourierbasis is deprecated; use make_fourierbasis_dm instead.",
                  DeprecationWarning, stacklevel=2)
    return make_fourierbasis_dm(alpha=alpha, tndm=tndm)

def makegp_fourier(psr, prior, components, T=None, mean=None, fourierbasis=fourierbasis, common=[], exclude=['f', 'df'], name='fourierGP', noisedict={}):
    argspec = inspect.getfullargspec(prior)
    # ncomp handles whether components is int, dictionary (keyed off parameters), or array of Fourier modes
    ncomp = lambda arg: components[arg] if isinstance(components, dict) else \
                        components if isinstance(components, (int, np.integer)) else len(components)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({ncomp(arg)})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    f, df, fmat = fourierbasis(psr, components, T=T)

    # f, df = matrix.jnparray(f), matrix.jnparray(df)
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap
    priorfunc.type = getattr(prior, 'type', None)

    if callable(fmat):
        argspec = inspect.getfullargspec(fmat)
        fargmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({len(f) // 2})' if argspec.annotations.get(arg) == typing.Sequence else '')
                   for arg in argspec.args if arg not in ['f', 'df']]

        def fmatfunc(params):
            return fmat(*[params[arg] for arg in fargmap])
        fmatfunc.params = fargmap

    # fixed noise hyperparameter analysis: if all hyperparameters (prior params, plus any variable
    # design-matrix params such as a chromatic index) are supplied in noisedict,
    # evaluate the prior and design matrix once and return a cached ConstantGP.
    fixed_params = priorfunc.params + (fmatfunc.params if callable(fmat) else [])
    if noisedict and mean is None and all(par in noisedict for par in fixed_params):
        phi = priorfunc(noisedict)
        fmat_const = fmatfunc(noisedict) if callable(fmat) else fmat
        gp = matrix.ConstantGP(matrix.NoiseMatrix12D_novar(phi), fmat_const)
    else:
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

    # ncomp handles whether components is int, dictionary (keyed off parameters), or array of Fourier modes
    ncomp = lambda arg: components[arg] if isinstance(components, dict) else \
                        components if isinstance(components, (int, np.integer)) else len(components)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in exclude]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({ncomp(arg)})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in exclude]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T=T) for psr in psrs])
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
    _, _, fmat = fourierbasis(psr, components, T=T)
    argname = f'{psr.name}_{name}_mean({fmat.shape[1]})'

    Fmat = matrix.jnparray(fmat)

    def delayfunc(params):
        return matrix.jnp.dot(Fmat, params[argname])
    delayfunc.params = [argname]

    return delayfunc

def makegp_fourier_variance(psr, components, T=None, name='fourierGP', noisedict={}):
    _, _, fmat = fourierbasis(psr, components, T=T)
    argname = f'{psr.name}_{name}_variance({fmat.shape[1]},{fmat.shape[1]})'

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
    # ncomp handles whether components is int, dictionary (keyed off parameters), or array of Fourier modes
    ncomp = components if isinstance(components, (int, np.integer)) else len(components)

    argspec = inspect.getfullargspec(prior)
    argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                (f'({ncomp})' if argspec.annotations.get(arg) == typing.Sequence else '')
                for arg in argspec.args if arg not in ['f', 'df']] for psr in psrs]

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T=T) for psr in psrs])
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

    gp.index = {f'{psr.name}_{name}_coefficients({2*ncomp})':
                slice((2*ncomp)*i, (2*ncomp)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp


def makeglobalgp_fourier(psrs, priors, orfs, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'],
                         name='fourierGlobalGP', meansname='meanFourierGlobalGP'):
    # ncomp handles whether components is int, dictionary (keyed off parameters), or array of Fourier modes
    ncomp = components if isinstance(components, (int, np.integer)) else len(components)

    priors = priors if isinstance(priors, list) else [priors]
    orfs   = orfs   if isinstance(orfs, list)   else [orfs]

    argmaps = []
    for prior, orf in zip(priors, orfs):
        argspec = inspect.getfullargspec(prior)
        priorname = f'{name}' if len(priors) == 1 else f'{name}_{re.sub("_", "", orf.__name__)}'
        argmaps.append([f'{priorname}_{arg}' + (f'({ncomp})' if argspec.annotations.get(arg) == typing.Sequence else '')
                        for arg in argspec.args if arg not in exclude])

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T=T) for psr in psrs])
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

def timeinterpbasis(psr, components, modes=None, T=None, start_time=None):
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
    def timeinterpbasis(psr, components, modes=None, T=None):
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

    def timeinterpbasis_dm(psr, nmodes, T=None):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T=T)
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

    def timeinterpbasis_chrom(psr, nmodes, T=None):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T=T)
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

    def dmbasis(psr, components, modes=None, T=None, fref=1400.0):
        t_coarse, dt_coarse, Bmat = basis(psr, components, T=T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return t_coarse, dt_coarse, Bmat * Dm[:, None]

    return dmbasis

def psd2cov(psdfunc, components, T, oversample=3, fmax_factor=1, cutoff=1):
    if not (isinstance(oversample, int) and isinstance(fmax_factor, int) and isinstance(cutoff, int)):
        raise ValueError('psd2cov: oversample, fmax_factor and cutoff must be integers.')

    if components % 2 == 0:
        raise ValueError('psd2cov: number of components must be odd.')

    scaled_components = (components - 1) * fmax_factor + 1
    n_freqs = int((scaled_components - 1) / 2 * oversample + 1)
    fmax = (scaled_components - 1) / T / 2
    freqs = np.linspace(0, fmax, n_freqs)
    df = 1 / T / oversample

    if cutoff is not None:
        i_cutoff = int(np.ceil(oversample / cutoff))
        fs, zs = matrix.jnparray(freqs[i_cutoff:]), jnp.zeros(i_cutoff)
    else:
        fs = matrix.jnparray(freqs)

    def covmat(*args):
        if cutoff is not None:
            psd = jnp.concatenate([zs, psdfunc(fs, 1.0, *args[2:])])
        else:
            psd = psdfunc(fs, 1.0, *args[2:])

        fullpsd = jnp.concatenate((psd, psd[-2:0:-1]))
        Cfreq = jnp.fft.ifft(fullpsd, norm='backward')
        Ctau = Cfreq.real * len(fullpsd) * df / 2

        return matrix.jsp.linalg.toeplitz(Ctau[:scaled_components:fmax_factor])
    covmat.__signature__ = inspect.signature(psdfunc)
    covmat.type = jax.Array

    return covmat

def makegp_fftcov(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, fourierbasis=None, common=[], name='fftcovGP', noisedict={}):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T=T,
                          fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                          common=common, name=name, noisedict=noisedict)

def makegp_fftcov_dm(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='dm_gp', fref=1400.0, noisedict={}):
    """FFT-covariance (time-domain) GP for DM noise (fixed chromatic index alpha = 2).

    DM counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation basis
    is replaced by :func:`make_timeinterpbasis_dm`, scaling each row by the cold-plasma
    dispersion factor ``(fref / psr.freqs) ** 2``. ``prior`` is a power-spectral-density
    function (e.g. :func:`powerlaw`) that is converted to a time-domain covariance via
    :func:`psd2cov`. For a free chromatic index use :func:`makegp_fftcov_chrom`.

    Passing a ``noisedict`` with all the prior hyperparameters (fixed hyperparameter analysis)
    returns a cached :class:`matrix.ConstantGP` instead of a :class:`matrix.VariableGP`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_dm(start_time=t0, order=order, fref=fref), common=common, name=name, noisedict=noisedict)

def makegp_fftcov_chrom(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='chrom_gp', fref=1400.0, noisedict={}):
    """FFT-covariance (time-domain) GP for chromatic noise with a variable index.

    Chromatic counterpart of :func:`makegp_fftcov`: the achromatic time-interpolation
    basis is replaced by :func:`make_timeinterpbasis_chromatic`, scaling each row by
    ``(fref / psr.freqs) ** alpha`` with the chromatic index ``alpha`` a free parameter.
    ``prior`` is a power-spectral-density function (e.g. :func:`powerlaw`) converted to a
    time-domain covariance via :func:`psd2cov`. For the alpha = 2 (DM) case use
    :func:`makegp_fftcov_dm`.

    Passing a ``noisedict`` with all the prior hyperparameters and the chromatic index
    (fixed hyperparameter analysis) returns a cached :class:`matrix.ConstantGP` instead of a
    :class:`matrix.VariableGP`.
    """
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_chromatic(start_time=t0, order=order, fref=fref), common=common, name=name, noisedict=noisedict)

def makecommongp_fftcov(psrs, prior, components, T, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, fourierbasis=None, common=[], vector=False, name='fftcovCommonGP'):
    return makecommongp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T,
                                fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                                common=common, vector=vector, name=name)

def makeglobalgp_fftcov(psrs, prior, orf, components, T, t0, order=1, oversample=3, fmax_factor=1, cutoff=1, fourierbasis=None, name='fftcovGlobalGP'):
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

def powerlaw_cutoff(f, df, log10_A, gamma, Nfreq_cutoff, *, tau=1.0):
    if tau <= 0:
        raise ValueError('powerlaw_cutoff: tau must be > 0.')
    mode_index = (jnp.arange(f.shape[0], dtype=jnp.float64) // 2) + 1.0
    gate = jax.nn.sigmoid((Nfreq_cutoff - mode_index + 0.5) / tau)
    return powerlaw(f, df, log10_A, gamma) * gate + 1e-15 # regularization

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
