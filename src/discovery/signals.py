import os
import re
import inspect
import typing
from collections.abc import Iterable

import numpy as np
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


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, selection=selection_backend_flags):
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

    if all(par in noisedict for par in params):
        if tnequad:
            noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        return matrix.NoiseMatrix1D_novar(noise)
    else:
        toaerrs, masks = matrix.jnparray(scale * psr.toaerrs), [matrix.jnparray(mask) for mask in masks]
        if tnequad:
            def getnoise(params):
                return sum(mask * (params[efac]**2 * toaerrs**2 + 10.0**(2 * (logscale + params[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            def getnoise(params):
                return sum(mask * params[efac]**2 * (toaerrs**2 + 10.0**(2 * (logscale + params[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))
        getnoise.params = params

        return matrix.NoiseMatrix1D_var(getnoise)

# ECORR quantization
#
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
def makegp_ecorr(psr, noisedict={}, enterprise=False, scale=1.0, selection=selection_backend_flags):
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

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umatall)

# timing model

def makegp_improper(psr, fmat, constant=1.0e40, name='improperGP', variable=False):
    if variable:
        def getphi(params):
            return constant * jnp.ones(fmat.shape[1])
        getphi.params = []

        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
    else:
        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)

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

    gp = makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable)
    gp.name = psr.name
    return gp

# Fourier GP

def getspan(psrs):
    if isinstance(psrs, Iterable):
        return max(psr.toas.max() for psr in psrs) - min(psr.toas.min() for psr in psrs)
    else:
        return psrs.toas.max() - psrs.toas.min()


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

def dmfourierbasis(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    Dm = (fref / psr.freqs)**2

    return f, df, fmat * Dm[:, None]

def dmfourierbasis_alpha(psr, components, T=None, fref=1400.0):
    f, df, fmat = fourierbasis(psr, components, T)

    fmat, fnorm = matrix.jnparray(fmat), matrix.jnparray(fref / psr.freqs)
    def fmatfunc(alpha):
        return fmat * fnorm[:, None]**alpha

    return f, df, fmatfunc

def makegp_fourier(psr, prior, components, T=None, fourierbasis=fourierbasis, common=[], name='fourierGP'):
    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in ['f', 'df']]

    # we'll create frequency bases using the longest vector parameter (e.g., for makefreespectrum_crn)
    if isinstance(components, dict):
        components = max(components.values())

    f, df, fmat = fourierbasis(psr, components, T)

    f, df = matrix.jnparray(f), matrix.jnparray(df)
    def priorfunc(params):
        return prior(f, df, *[params[arg] for arg in argmap])
    priorfunc.params = argmap

    # TODO: I'd like my makegp_fourier to be cleaner than this
    # also, the argmap code can be modularized
    if callable(fmat):
        argspec = inspect.getfullargspec(fmat)
        fargmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                   (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                   for arg in argspec.args if arg not in ['f', 'df']]

        def fmatfunc(params):
            return fmat(*[params[arg] for arg in fargmap])
        fmatfunc.params = fargmap

    gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(priorfunc), fmatfunc if callable(fmat) else fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(0,len(f))} # better for cosine
    gp.name, gp.pos = psr.name, psr.pos
    gp.gpname, gp.gpcommon = name, common

    return gp


# for use in ArrayLikelihood. Same process for all pulsars.
def makecommongp_fourier(psrs, prior, components, T, fourierbasis=fourierbasis, common=[], vector=False, name='fourierCommonGP'):
    argspec = inspect.getfullargspec(prior)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in ['f', 'df']]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in ['f', 'df']]

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
    else:
        vprior = jax.vmap(prior, in_axes=[None, None] +
                                         [0 if isinstance(argmap, list) else None for argmap in argmaps])

        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(f, df, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(priorfunc), fmats)
    gp.index = {f'{psr.name}_{name}_coefficients({2*components})': slice(2*components*i,2*components*(i+1))
                for i, psr in enumerate(psrs)}

    return gp

# component-wise GP

# def makegp_fourier_components(psr, prior, components, T=None, fourierbasis=fourierbasis, common=[], name='fourierGP'):
#     argspec = inspect.getfullargspec(prior)
#     argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
#               (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
#               for arg in argspec.args if arg not in ['f', 'df']]

#     argname = f'{psr.name}_{name}_coefficients({components*2})'

#     f, df, fmat = fourierbasis(psr, components, T)

#     f, df = matrix.jnparray(f), matrix.jnparray(df)
#     def priorfunc(params):
#         return prior(f, df, *[params[arg] for arg in argmap])
#     priorfunc.params = argmap

#     def componentfunc(params):
#         return params[argname]
#     componentfunc.params = [argname]

#     Fmat = matrix.jnparray(fmat)

#     return matrix.ComponentGP(matrix.NoiseMatrix1D_var(priorfunc), Fmat, componentfunc)

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

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix1D_var(priorfunc), fmats, invprior)
    gp.index = {f'{psr.name}_{name}_coefficients({2*components})':
                slice((2*components)*i, (2*components)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp

def makegp_rngw_global(psrs, rnprior, rncomponents, gwprior, gworf, gwcomponents, T, name='red_noise'):
    gwargspec = inspect.getfullargspec(gwprior)
    gwargmap  =  [f'gw_{arg}' + (f'({gwcomponents})' if gwargspec.annotations.get(arg) == typing.Sequence else '')
                  for arg in gwargspec.args if arg not in ['f','df']]

    rnargspec = inspect.getfullargspec(rnprior)
    rnargmaps = [[f'{psr.name}_{name}_{arg}' + (f'({rncomponents})' if rnargspec.annotations.get(arg) == typing.Sequence else '')
                  for arg in rnargspec.args if arg not in ['f','df']]
                 for psr in psrs]

    # assume rncomponents > gwcomponents
    fs, dfs, fmats = zip(*[fourierbasis(psr, rncomponents, T) for psr in psrs])
    f, df = fs[0], dfs[0]

    gworfmat = matrix.jnparray([[gworf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs])
    gwmask = matrix.jnparray(np.arange(2*rncomponents) < 2*gwcomponents)

    diagrange = matrix.intarray(range(2*rncomponents*len(psrs)))

    def priorfunc(params):
        gwphidiag = gwmask * gwprior(f, df, *[params[arg] for arg in gwargmap])
        Phi = jnp.block([[jnp.diag(jnp.dot(gwphidiag, val)) for val in row] for row in gworfmat])

        rnphidiag = jnp.concatenate([rnprior(f, df, *[params[arg] for arg in argmap]) for argmap in rnargmaps])
        Phi = Phi.at[(diagrange, diagrange)].add(rnphidiag)

        return Phi
    priorfunc.params = gwargmap + sum(rnargmaps, [])

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix2D_var(priorfunc), fmats, None)
    gp.index = {f'{psr.name}_{name}_coefficients({2*rncomponents})':
                slice((2*rncomponents)*i, (2*rncomponents)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp

def makegp_fourier_global(psrs, priors, orfs, components, T, fourierbasis=fourierbasis, name='globalFourierGP'):
    priors = priors if isinstance(priors, list) else [priors]
    orfs   = orfs   if isinstance(orfs, list)   else [orfs]

    argmaps = []
    for prior, orf in zip(priors, orfs):
        argspec = inspect.getfullargspec(prior)
        priorname = f'{name}' if len(priors) == 1 else f'{name}_{re.sub("_", "", orf.__name__)}'
        argmaps.append([f'{priorname}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                        for arg in argspec.args if arg not in ['f', 'df']])

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])

    orfmats = [matrix.jnparray([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs]) for orf in orfs]

    if len(priors) == 1 and len(orfs) == 1:
        prior, orfmat, argmap = priors[0], orfmats[0], argmaps[0]

        def priorfunc(params):
            phidiag = prior(f, df, *[params[arg] for arg in argmap])

            # the jnp.dot handles the "pixel basis" case where the elements of orfmat are n-vectors
            # and phidiag is an (m x n)-matrix; here n is the number of pixels and m of Fourier components
            return jnp.block([[jnp.diag(jnp.dot(phidiag, val)) for val in row] for row in orfmat])
        priorfunc.params = argmap

        # if we're not in the pixel-basis case we can take a shortcut in making the inverse
        if orfmat.ndim == 2:
            # invorf, orflogdet = jnp.linalg.inv(orfmat), jnp.linalg.slogdet(orfmat)[1]
            invorf, orflogdet = matrix.jnparray(np.linalg.inv(orfmat)), np.linalg.slogdet(orfmat)[1]
            def invprior(params):
                invphidiag = 1.0 / prior(f, df, *[params[arg] for arg in argmap])

                # |S_ij Gamma_ab| = prod_i (|S_i Gamma_ab|) = prod_i (S_i^npsr |Gamma_ab|)
                # log |S_ij Gamma_ab| = log (prod_i S_i^npsr) + log prod_i |Gamma_ab|
                #                     = npsr * sum_i log S_i + nfreqs |Gamma_ab|

                return (jnp.block([[jnp.diag(val * invphidiag) for val in row] for row in invorf]),
                        invphidiag.shape[0] * orflogdet - orfmat.shape[0] * jnp.sum(jnp.log(invphidiag)))
            invprior.params = argmap
        else:
            invprior = None
    else:
        def priorfunc(params):
            phidiags = [prior(f, df, *[params[arg] for arg in argmap]) for prior, argmap in zip(priors, argmaps)]

            return sum(jnp.block([[jnp.diag(val * phidiag) for val in row] for row in orfmat])
                       for phidiag, orfmat in zip(phidiags, orfmats))
        priorfunc.params = sorted(set.union(*[set(argmap) for argmap in argmaps]))

        invprior = None

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix2D_var(priorfunc), fmats, invprior)
    gp.index = {f'{psr.name}_{name}_coefficients({2*components})':
                slice((2*components)*i, (2*components)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp


datadir = os.path.join(os.path.dirname(__file__), '../../data')

cosinet_g = np.linspace(0, 7, 71)
cosinet_t = np.linspace(0, 1, 100)
try:
    cosinet_c = np.load(os.path.join(datadir, 'cosine_powerlaw_tb.npy'))
except:
    pass

import functools
interp_gammas = jax.vmap(jnp.interp, in_axes=(None, None, 1))

# interp_taus  = jax.vmap(jax.vmap(functools.partial(jnp.interp, left=0.0, right=0.0),
#                                  in_axes=(0, None, None)),
#                         in_axes=(0, None, None))

interp_bound = lambda x, xp, vp, r: jnp.interp(x, xp, vp, right=r, left=0.0)
interp_taus = jax.vmap(jax.vmap(interp_bound, in_axes=(0, None, None, None)), in_axes=(0, None, None, None))

def makepowerlaw_timedomain(Tspan):
    T = Tspan

    def powerlaw(tau, log10_A, gamma):
        norm = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * T**(gamma - 1.0)

        intmap = interp_gammas(gamma, cosinet_g, cosinet_c)
        intval = interp_taus(tau / T, cosinet_t, intmap, 1/norm)

        return norm * intval

    return powerlaw

def makepowerlaw_crn_timedomain(Tspan, Tspan_crn=None):
    get_tmat = makepowerlaw_timedomain(Tspan)
    get_tmat_crn = makepowerlaw_timedomain(Tspan if Tspan_crn is None else Tspan_crn)

    def powerlaw(tau, log10_A, gamma, crn_log10_A, crn_gamma):
        return get_tmat(tau, log10_A, gamma) + get_tmat_crn(tau, crn_log10_A, crn_gamma)

    return powerlaw


def makegp_timedomain(psr, covariance, dt=1.0, common=[], name='timedomainGP'):
    argspec = inspect.getfullargspec(covariance)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args if arg not in ['tau']]

    bins = quantize(psr.toas, dt)
    Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
    toas = psr.toas @ Umat / Umat.sum(axis=0)

    get_tmat = covariance
    tau = jnp.abs(toas[:, jnp.newaxis] - toas[jnp.newaxis, :])

    def getphi(params):
        return get_tmat(tau, *[params[arg] for arg in argmap])
    getphi.params = argmap

    return matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), Umat)


def makecommongp_timedomain(psrs, covariance, dt=1.0, common=[], name='timedomainCommonGP'):
    argspec = inspect.getfullargspec(covariance)
    argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
                for psr in psrs] for arg in argspec.args if arg not in ['tau']]

    # quantize toas for each pulsar and create "exploder" U matrices
    def quantized(psr):
        bins = quantize(psr.toas, dt)
        Umat = np.vstack([bins == i for i in range(bins.max() + 1)]).T.astype('d')
        return psr.toas @ Umat / Umat.sum(axis=0), Umat
    toas, Umats = zip(*[quantized(psr) for psr in psrs])

    # pad the Umats and toas to the same number of coarse toas
    nepochs = max(len(toa) for toa in toas)
    Umats = [np.pad(Umat, ((0,0), (0,nepochs - Umat.shape[1]))) for Umat in Umats]
    stdtoas = np.array([np.pad(toa, (0,nepochs - len(toa))) for toa in toas])

    taus = np.abs(stdtoas[:, :, jnp.newaxis] - stdtoas[:, jnp.newaxis, :])
    # the idea is to manage the padded region by triggering the left interp to get 0,
    # and the right interp to get 1 / diagonal value, which becomes one with normalization
    # the resulting matrix is poorly conditioned
    for i, toa in enumerate(toas):
        taus[i, len(toa):, :] = -1.0
        taus[i, :, len(toa):] = -1.0
        taus[i, range(len(toa), nepochs), range(len(toa), nepochs)] = 1e40

    get_tmat = jax.vmap(covariance, in_axes=[0] + [0]*len(argmaps))

    def getphi(params):
        vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                for argmap in argmaps]
        return get_tmat(taus, *vpars)
    getphi.params = sorted(set(sum([argmap for argmap in argmaps], [])))

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix2D_var(getphi), Umats)

    return gp


def powerlaw(f, df, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df

def make_powerlaw(scale=1.0):
    logscale = np.log10(scale)

    def powerlaw(f, df, log10_A, gamma):
        logpl = (2.0 * log10_A) - jnp.log10(12.0 * jnp.pi**2) + (gamma - 3.0) * jnp.log10(const.fyr) - gamma * jnp.log10(f) + jnp.log10(df)
        return 10**(2*logscale + logpl)

    return powerlaw

def powerlaw_no_df(f, log10_A, gamma):
    return (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)

def freespectrum(f, df, log10_rho: typing.Sequence):
    return jnp.repeat(10.0**(2.0 * log10_rho), 2)


# combined red_noise + crn

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

def makepowerlaw_crn_samedim_const(crn_gamma='variable'):
    def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_const):
        phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
        phi = phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * \
                    const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) * df + \
                    (10.0**(2.0 * crn_log10_const)) / 12.0 / jnp.pi**2 * \
                    const.fyr ** (- 3.0) * df
        return phi
    
    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn

def makepowerlaw_crn_samedim_broken_powerlaw(**constants):
    def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb):
        #crn_log10_fb: log10 transition frequency at which slope switches from gamma to zero
        kappa=0.1 # smoothness of transition
        phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
        phi = phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) *\
             f ** (-crn_gamma) * df * (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)
        return phi
    
    constants = {varname: value for varname, value in constants.items()
                                if value != 'variable'}
    if constants:
        return matrix.partial(powerlaw_crn, **constants)
    else:
        return powerlaw_crn


# combined red_noise + crn FFT

def makepowerlaw_crn_fft(components_crn, oversample_crn=None, crn_gamma='variable'):

    if oversample_crn is None:
        oversample_crn = 2

    n_freqs_crn = (components_crn//2 + 1 ) * oversample_crn
    if matrix.jnp == jnp:
        def powerlaw_crn(f, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)
            phi = phi.at[:n_freqs_crn].add((10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 *
                                            const.fyr ** (crn_gamma - 3.0) * f[:n_freqs_crn] ** (-crn_gamma))
            return phi
    elif matrix.jnp == np:
        def powerlaw_crn(f, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / np.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)
            phi[:n_freqs_crn] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / np.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:n_freqs_crn] ** (-crn_gamma))
            return phi

    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn
    
def makepowerlaw_crn_fft_samedim(crn_gamma='variable'):
    def powerlaw_crn(f, log10_A, gamma, crn_log10_A, crn_gamma):
        phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)
        phi = phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * \
                                        const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma)
        return phi
    
    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn
    
def makepowerlaw_crn_fft_samedim_const(crn_gamma='variable'):
    def powerlaw_crn(f, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_const):
        phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)
        phi = phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * \
                    const.fyr ** (crn_gamma - 3.0) * f ** (-crn_gamma) + \
                    (10.0**(2.0 * crn_log10_const)) / 12.0 / jnp.pi**2 * \
                    const.fyr ** (- 3.0)
        return phi
    
    if crn_gamma != 'variable':
        return matrix.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn

def makepowerlaw_crn_fft_samedim_broken_powerlaw(**constants):
    def powerlaw_crn(f, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb):
        #crn_log10_fb: log10 transition frequency at which slope switches from gamma to zero
        kappa=0.1 # smoothness of transition
        phi = (10.0**(2.0 * log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma)
        phi = phi + (10.0**(2.0 * crn_log10_A)) / 12.0 / jnp.pi**2 * const.fyr ** (crn_gamma - 3.0) *\
             f ** (-crn_gamma) * (1 + (f / 10**crn_log10_fb) ** (1 / kappa)) ** (kappa * crn_gamma)
        return phi
    
    constants = {varname: value for varname, value in constants.items()
                                if value != 'variable'}
    if constants:
        return matrix.partial(powerlaw_crn, **constants)
    else:
        return powerlaw_crn


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


# delay

def makedelay(psr, delay, common=[], name='delay'):
    argspec = inspect.getfullargspec(delay)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}')
              for arg in argspec.args]

    def delayfunc(params):
        return delay(*[params[arg] for arg in argmap])
    delayfunc.params = argmap

    return delayfunc

# standard parameters t, pos, d;
def makedelay_deterministic(psr, delay, name='deterministic'):
    argspec = inspect.getfullargspec(prior)
    argmap = [f'{name}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in ['t', 'pos', 'd']]

    def delayfunc(params):
        return delay(t, pos, d, *[params[arg] for arg in argmap])
    delayfunc.params = argmap

    return delayfunc

##### FFT METHOD #####

def build_piecewise_linear_B(t_fine, t_coarse):
    """
    Build matrix B of shape (len(t_fine), len(t_coarse)) for piecewise-linear interpolation.
    That is, each row i corresponds to t_fine[i]; each column j is the hat function phi_j
    that is 1 at t_coarse[j] and 0 at t_coarse[j-1], t_coarse[j+1].

    In the naive matrix approach, we do:
      B[i,j] = phi_j(t_fine[i])
    and phi_j(t) is the linear function that is:
      - 0 for t < t_{j-1} or t > t_{j+1}
      - up to 1 at t_j
    We'll handle boundaries carefully.

    Complexity: O(N_fine * N_coarse).
    """
    Nf = len(t_fine)
    Nc = len(t_coarse)
    B = np.zeros((Nf, Nc), dtype=float)

    # We'll do a naive approach: for each i in [0..Nf-1], find which segment [t_j, t_{j+1}] it falls into.
    # Then B[i,j], B[i,j+1] are the only nonzero entries (besides boundary cases).
    # That is O(Nf log Nc) if we do a binary search for each t_fine[i], or O(Nf + Nc) if we do a single sweep.
    # For now, we do a single pass with two pointers.

    j = 0  # pointer for coarse intervals
    for i in range(Nf):
        tf = t_fine[i]

        # move j so that t_coarse[j] <= tf < t_coarse[j+1], except boundary
        while j < Nc-2 and tf > t_coarse[j+1]:
            j += 1

        # Now the point t_fine[i] is between t_coarse[j] and t_coarse[j+1].
        if j == Nc-1:
            # We are beyond the last coarse point
            # so we just clamp. phi_{N-1}(t) = 1 at t_{N-1}.
            B[i, Nc-1] = 1.0
        else:
            # in [t_j, t_{j+1}]
            tj = t_coarse[j]
            tjp1 = t_coarse[j+1]
            # the distance
            length = tjp1 - tj
            if length <= 0:
                # degenerate
                B[i,j] = 1.0
            else:
                # linear fraction from j to j+1
                frac = (tf - tj)/length
                # phi_j(t) = 1 - frac
                # phi_{j+1}(t) = frac
                B[i,j]   = 1.0 - frac
                B[i,j+1] = frac

    return B

def coarse_grained_basis(psr, components, start_time=None, T=None):
    if T is None:
        T = getspan(psr)
    if start_time is None:
        start_time = jnp.min(psr.toas)

    t_coarse = jnp.linspace(start_time, start_time+T, num=components)
    dt = t_coarse[2] - t_coarse[1]
    t_fine = psr.toas
    B = build_piecewise_linear_B(t_fine, t_coarse)

    return t_coarse, dt, B


# The FFT functions
def psd_to_covariance(df, psd):
    """
    Convert a one-sided PSD (f >= 0) into the time-domain covariance function C(tau).

    Parameters
    ----------
    df   : df.
    psd : 1D array, shape (N,)
          One-sided PSD values, psd[k] = S(f[k]).
          Must be real and >= 0 (for a valid PSD), though not enforced here.

    Returns
    -------
    #tau : 1D array, shape (M = 2*N - 2,)
    #      Time lags [seconds], running from 0 to positive.
    Ctau: 1D array, shape (M = 2*N - 2,)
          Covariance function samples, C(tau).
          Real-valued.
    """

    N = len(psd)
    if N < 2:
        raise ValueError("Need at least two frequency points in f")

    psd_mirror = psd[1:-1][::-1]                    # shape (N-2,)
    full_psd = jnp.concatenate((psd, psd_mirror))    # shape (2N - 2,)
    M = len(full_psd)

    c_freq = jnp.fft.ifft(full_psd, norm='backward')
    Ctau = c_freq * M * df / 2

    return Ctau.real

def psd_to_covfunc(psdfunc, T, n_modes, over_sample, n_fL, *params):
    """
    Given a frequency over-sample rate, number of course-grained times, and T
    select the number of frequencies, fmax, delta_f, and
    turn a PSD function into a covariance function
    """

    if n_modes%2==0:
        raise ValueError("Number of coarse time-samples needs to be odd")

    n_freqs = (n_modes//2 + 1 ) * over_sample
    fmax = (n_modes - 1)/(2*T)
    freqs = jnp.linspace(0, fmax, n_freqs)
    df = fmax/(n_freqs-1)
    
    ind = int(np.ceil(1/(n_fL*T)/df))

    psd = psdfunc(freqs[ind:], *params)
    psd = jnp.concatenate([jnp.zeros(ind), psd])
    Ctau = psd_to_covariance(df, psd)

    return Ctau[:n_modes]

def ctau_to_cnm(ctau):
    """Build the matrix Cnm = C(tau[n] - tau[m]) given:
      - tau is 1D, uniformly spaced
      - ctau[i] = C(tau[i]),
      - no interpolation is needed because of regular grid
    """

    N=len(ctau)
    inds = jnp.arange(0,N)
    n = inds[:,None]
    m = inds[None,:]
    nm = jnp.abs(n-m)

    return ctau[nm]

def psd_to_cnm(psdfunc, T, n_modes, over_sample, n_fL, *params):
    """Take a psd function, and turn it into a coarse
    time-domain covariance matrix

    Returns C(np.abs(u[:,None]-u[None,:]), u
    """

    covfunc = psd_to_covfunc(psdfunc, T, n_modes, over_sample, n_fL, *params)

    return ctau_to_cnm(covfunc)


def makegp_coarse_grained(psr, prior, components, oversample=None, cutoff=None, start_time=None, T=None,
                          basis_function=coarse_grained_basis, common=[], name='coarsegrainGP'):

    argspec = inspect.getfullargspec(prior)
    argmap = [(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
              (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '')
              for arg in argspec.args if arg not in ['f']]

    # we'll create coarse grained bases using the longest vector parameter
    if isinstance(components, dict):
        components = max(components.values())

    if components%2==0:
        raise ValueError("Number of coarse time-samples needs to be odd")

    if oversample is None:
        oversample = 2
    if cutoff is None:
        cutoff = oversample + 1

    t_coarse, _ , Bmat = basis_function(psr, components, start_time=start_time, T=T)

    if T is None:
        T = getspan(psr)
    if start_time is None:
        start_time = jnp.min(psr.toas)

    def priorfunc(params):
        C_nm = psd_to_cnm(prior, T, components, oversample, cutoff, *[params[arg] for arg in argmap])
        return C_nm
    priorfunc.params = argmap

    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(priorfunc), Bmat)
    gp.index = {f'{psr.name}_{name}_coefficients({t_coarse.size})': slice(0, t_coarse.size)}
    gp.name, gp.pos = psr.name, psr.pos
    gp.gpname, gp.gpcommon = name, common

    return gp


def makecommongp_coarse_grained(psrs, prior, components, oversample=None, cutoff=None, start_time=None, T=None, basis_function=coarse_grained_basis, common=[], name='coarsegrainCommonGP', vector=False):
    argspec = inspect.getfullargspec(prior)

    if vector:
        argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{name}_{arg}({len(psrs)})' for arg in argspec.args if arg not in ['f']]
    else:
        argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
                    (f'({components[arg] if isinstance(components, dict) else components})' if argspec.annotations.get(arg) == typing.Sequence else '') for psr in psrs]
                   for arg in argspec.args if arg not in ['f']]


    if isinstance(components, dict):
        components = max(components.values())

    if components%2==0:
        raise ValueError("Number of coarse time-samples needs to be odd")

    if oversample is None:
        oversample = 2
    if cutoff is None:
        cutoff = oversample + 1

    if T is None:
        T = getspan(psrs)
    if start_time is None:
        start_time = min(psr.toas.min() for psr in psrs)


    t_coarse, _ , Bmats = zip(*[basis_function(psr, components, start_time=start_time, T=T) for psr in psrs])
    t_coarse = t_coarse[0]
    # fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    # f, df = fs[0], dfs[0]

    if vector:
        vpsd_to_cnm = jax.vmap(psd_to_cnm, in_axes=[None]*5 +
                                         [0 if f'({len(psrs)})' in arg else None for arg in argmap])

        def priorfunc(params):
            return vpsd_to_cnm(prior, T, components, oversample, cutoff, *[params[arg] for arg in argmap])

        priorfunc.params = sorted(argmap)
    else:
        # vprior = jax.vmap(prior, in_axes=[None, None] +
        #                                  [0 if isinstance(argmap, list) else None for argmap in argmaps])
        vpsd_to_cnm = jax.vmap(psd_to_cnm, in_axes=[None]*5 +
                                         [0 if isinstance(argmap, list) else None for argmap in argmaps])
        def priorfunc(params):
            vpars = [matrix.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vpsd_to_cnm(prior, T, components, oversample, cutoff, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))

    gp = matrix.VariableGP(matrix.VectorNoiseMatrix2D_var(priorfunc), Bmats)
    gp.index = {f'{psr.name}_{name}_coefficients({t_coarse.size})': slice(t_coarse.size*i,t_coarse.size*(i+1))
                for i, psr in enumerate(psrs)}

    return gp
