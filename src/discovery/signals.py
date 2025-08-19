import os
import re
import inspect
import types
import typing
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
def makenoise_measurement_simple(psr, noisedict={}):
    efac = f'{psr.name}_efac'
    log10_t2equad = f'{psr.name}_log10_t2equad'
    params = [efac, log10_t2equad]

    if all(par in noisedict for par in params):
        noise = noisedict[efac]**2 * (psr.toaerrs**2 + 10.0**(2.0 * noisedict[log10_t2equad]))

        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs = matrix.jnparray(psr.toaerrs)
        def getnoise(params):
            return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)


# nanograv backends
def selection_backend_flags(psr):
    return psr.backend_flags


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, ecorr=False, selection=selection_backend_flags, vectorize=True,
                          outliers=False):
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
            egp = makegp_ecorr(psr, noisedict=noisedict, enterprise=True, scale=scale, selection=selection)
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
            egp = makegp_ecorr(psr, noisedict={}, enterprise=True, scale=scale, selection=selection)
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
        getphi.params = Params

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umat)

# nanograv backends
def makegp_ecorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=selection_backend_flags, name='ecorrGP'):
    log10_ecorrs, Umats = [], []

    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]
    for backend, mask in zip(backends, masks):
        log10_ecorrs.append(f'{psr.name}_{backend}_log10_ecorr')

        # TOAs that do not belong to this mask get index zero, which is ignored below.
        # This will fail if there's only one mask that covers all TOAs
        bins = quantize(psr.toas * mask)

        if enterprise:
            # legacy accounting of degrees of freedom
            uniques, counts = np.unique(quantize(psr.toas * mask), return_counts=True)
            Umats.append(np.vstack([bins == i for i, cnt in zip(uniques[1:], counts[1:]) if cnt > 1]).T)
        else:
            Umats.append(np.vstack([bins == i for i in range(1, bins.max() + 1)]).T)
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
        def getphi(params):
            return params[f"{psr.name}_{name}_coefficients({fmat.shape[1]})"]

        getphi.params = [f"{psr.name}_{name}_coefficients({fmat.shape[1]})"]
        gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    else:
        gp = matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)

    gp.name = psr.name
    gp.gpname = name

    return gp

def makegp_timing(psr, constant=1.0e40, variance=None, svd=False, scale=1.0, variable=False, timing_inds=None):
    # Set prior for marginalisation
    if variance is not None:
        constant = variance * psr.Mmat.shape[0] / psr.Mmat.shape[1]

    # Scale the design matrix
    if svd and timing_inds is None:
        fmat, _, _ = np.linalg.svd(scale * psr.Mmat, full_matrices=False)
    else:
        fmat = np.asarray(psr.Mmat / np.sqrt(np.sum(psr.Mmat**2, axis=0)), dtype=np.float64)

    # Case 1: variable=False, full analytical marginalisation.
    #       Return a constant GP.
    if not variable:
        return makegp_improper(psr, fmat, constant=constant, name='timingmodel_marg')

    # Case 2: variable=True and timing_inds=None, sampling all timing model coefficients.
    #       Return a variable GP.
    if timing_inds is None:
        return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=True)

    # Case 3: variable=True and timing_inds is not None, sampling only a subset of timing model coefficients.
    #       Force svd=False and return a constant GP plus a variable GP.
    if svd:
        print("Warning: `svd=True` ignored when sampling a subset of the raw timing model design matrix.")
        M = np.asarray(psr.Mmat) # Do not normalise the design matrix. Sample as-is

    timing_inds = jnp.asarray(timing_inds, dtype=int)
    mask = np.zeros(fmat.shape[1], dtype=bool)
    mask[timing_inds] = True

    fmat_marg = fmat[:, ~mask] # Marginalised design matrix
    gp_marg = makegp_improper(psr, fmat_marg, constant=constant, name='timingmodel_marg')

    fmat_var = M[:, mask] # Sampled design matrix
    coeffs = f"{psr.name}_timingmodel_coefficients({fmat_var.shape[1]})"
    def tm_delay(params):
        return jnp.dot(fmat_var, jnp.asarray(params[coeffs]))
    tm_delay.params = [coeffs]

    return [tm_delay, gp_marg]


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
        fmat[:, 2*i  ] = np.sin(2.0 * jnp.pi * f[i] * psr.toas)
        fmat[:, 2*i+1] = np.cos(2.0 * jnp.pi * f[i] * psr.toas)

    return np.repeat(f, 2), np.repeat(df, 2), fmat

def fourierbasis_dm(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    Dm = (fref / psr.freqs)**2

    return f, df, fmat * Dm[:, None]

def fourierbasis_chrom(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(alpha):
        return fmat * fnorm[:, None]**alpha

    return f, df, fmatfunc



def make_fourierbasis_chrom_inj(alpha):
    def fourierbasis_chrom_inj(psr, components, alpha, T=None, fref=1400.0):
        f, df, fmat = fourierbasis(psr, components, T)

        fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)

        chrom = (fref / psr.freqs)**alpha

        return f, df,  fmat * chrom[:, None]
    
	return fourierbasis_chrom_inj


def fourierbasis_band_low(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(fcutoff):
        scale = 5.0
        band_filter = jnp.reciprocal(1.0 + jnp.exp((psr.freqs - fcutoff) / scale))  # at scale=5.0 MHz, the 10% to 90% roll-off happens across ~22MHz.
        # band_filter = jnp.heaviside(fcutoff - psr.freqs, 0.0) # 0.0 for freqs >= fcutoff
        return fmat * fnorm[:, None] * band_filter[:, None]

    return f, df, fmatfunc

def fourierbasis_band_low_alpha(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(fcutoff, alpha):
        scale = 5.0
        band_filter = jnp.reciprocal(1.0 + jnp.exp((psr.freqs - fcutoff) / scale))  # at scale=5.0 MHz, the 10% to 90% roll-off happens across ~22MHz.
        # band_filter = jnp.heaviside(fcutoff - psr.freqs, 0.0) # 0.0 for freqs >= fcutoff
        return fmat * fnorm[:, None]**alpha * band_filter[:, None]

    return f, df, fmatfunc

def make_fourierbasis_dm(alpha=2.0, tndm=False): #  DM fourier basis shouldn't have alpha
    def basis(psr, components, T=None, fref=1400.0):
        f, df, fmat = fourierbasis(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        return f, df, fmat * Dm[:, None]

    return basis

def make_fourierbasis_chrom(alpha=4.0, tndm=False):
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

    f, df = matrix.jnparray(f), matrix.jnparray(df)
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
                   for arg in margs if not hasattr(psr, arg) and arg not in exclude}

        psrpars = {arg: getattr(psr, arg) for arg in margspec.args if hasattr(psr, arg)}

        def meanfunc(params):
            return mean(f, df, *psrpars.values(), **{arg: params[argname] for arg, argname in margmap.items()})
        meanfunc.params = sorted(margmap.values())

        gp.mean = meanfunc

    return gp


# for use in ArrayLikelihood. Same process for all pulsars.
def makecommongp_fourier(psrs, prior, components, T, fourierbasis=fourierbasis, common=[], exclude=['f', 'df'], vector=False, name='fourierCommonGP'):
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
        margmaps = [{arg: (f'{meansname}_{arg}' if f'{meansname}_{arg}' in common else f'{psr.name}_{meansname}_{arg}') +
                     (f'({components})' if (margspec.annotations.get(arg) == typing.Sequence and components is not None) else '')
                     for arg in margs if not hasattr(psr, arg) and arg not in exclude}
                    for psr in psrs]

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

def make_timeinterpbasis_chromatic(start_time=None, order=1, fref=1400.0):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_chrom(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs)
        def Bmat_func(alpha):
            return (scale[:, None]**alpha) * Bmat
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_chrom

def make_timeinterpbasis_band(start_time=None, order=1, fref=1400.0):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_band(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs)
        def Bmat_func(fcutoff):
            band_filter = jnp.heaviside(fcutoff - psr.freqs, 0.0) # 0.0 for freqs >= fcutoff
            return scale[:, None] * Bmat * band_filter[:, None]
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_band

def make_timeinterpbasis_band_alpha(start_time=None, order=1, fref=1400.0):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_band_alpha(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs)
        def Bmat_func(fcutoff, alpha):
            band_filter = jnp.heaviside(fcutoff - psr.freqs, 0.0) # 0.0 for freqs >= fcutoff
            return scale[:, None]**alpha * Bmat * band_filter[:, None]  # 0.0 for freqs >= fcutoff
        return t_coarse, dt_coarse, Bmat_func

    return timeinterpbasis_band_alpha

def make_timeinterpbasis_dm(start_time=None, order=1, fref=1400.0):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)

    def timeinterpbasis_dm(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        scale = (fref / psr.freqs) ** 2
        return t_coarse, dt_coarse, scale[:, None] * Bmat

    return timeinterpbasis_dm

def make_timeinterpbasis_solar(start_time=None, order=1):
    timeinterpbasis_achrom = make_timeinterpbasis(start_time=start_time, order=order)
    from .solar import theta_impact, dm_solar

    def timeinterpbasis_solar(psr, nmodes, T):
        t_coarse, dt_coarse, Bmat = timeinterpbasis_achrom(psr, nmodes, T)
        theta, R_earth, _, _ = theta_impact(psr)
        dm_sol_wind = dm_solar(1.0, theta, R_earth)
        dt_DM = dm_sol_wind * 4.148808e3 / (psr.freqs**2)
        return t_coarse, dt_coarse, dt_DM[:, None] * Bmat
    return timeinterpbasis_solar

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

def makegp_fftcov(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis(start_time=t0, order=order), common=common, name=name)

def makegp_fftcov_dm(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP_dm', fref=1400.0):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_dm(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_chrom(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP_chrom', fref=1400.0):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_chromatic(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_band(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP_chrom', fref=1400.0):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_band(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_band_alpha(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP_chrom', fref=1400.0):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_band_alpha(start_time=t0, order=order, fref=fref), common=common, name=name)

def makegp_fftcov_solar(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='fftcovGP_solar'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                          components, T=T, fourierbasis=make_timeinterpbasis_solar(start_time=t0, order=order), common=common, name=name)

def makecommongp_fftcov(psrs, prior, components, T, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], vector=False, name='fftcovCommonGP'):
    return makecommongp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff),
                                components, T, fourierbasis=make_timeinterpbasis(start_time=t0, order=order), common=common, vector=vector, name=name)

def makeglobalgp_fftcov(psrs, prior, orf, components, T, t0, order=1, oversample=3, fmax_factor=1, cutoff=1, name='fftcovGlobalGP'):
    return makeglobalgp_fourier(psrs, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), orf,
                                components, T, fourierbasis=make_timeinterpbasis(start_time=t0, order=order), name=name)


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
