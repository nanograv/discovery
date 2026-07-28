import os
import re
import inspect
import math as _math
import types
import typing
import warnings
from collections.abc import Iterable

import numpy as np
import scipy.interpolate as si
import jax
import jax.numpy as jnp

from . import utils
from . import _kernels as kernels
from . import const
from . import solar

# residuals

def residuals(psr):
    return psr.residuals


# EFAC/EQUAD/ECORR noise
#
# The measurement-noise constructors `makenoise_measurement` and
# `makenoise_measurement_simple` now live in `measurement_noise.py` (collapsed
# form -- no _novar/_var class enumeration; the variant is chosen by the
# `_kernels` factory). They are re-exported at the bottom of this module so
# `signals.makenoise_measurement` and `ds.makenoise_measurement` keep resolving.

# nanograv backends
def selection_backend_flags(psr):
    return psr.backend_flags


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

        return utils.ConstantGP(kernels.NoiseMatrix1D_novar(phi), Umat)
    else:
        ones = utils.jnparray(ones)
        def getphi(params):
            return (10.0**(2.0 * params[log10_ecorr])) * ones
        getphi.params = params

        return utils.VariableGP(kernels.NoiseMatrix1D_var(getphi), Umat)

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

            gp = utils.VariableGP(kernels.NoiseMatrix1D_var(getphi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []

            return gp
        else:
            gp = utils.ConstantGP(kernels.NoiseMatrix1D_novar(phi), Umatall)
            gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0, Umatall.shape[1])}
            gp.name, gp.pos = psr.name, psr.pos
            gp.gpname, gp.gpcommon = name, []
            return gp
    else:
        pmasks = [utils.jnparray(pmask) for pmask in pmasks]
        def getphi(params):
            return sum(10.0**(2 * (logscale + params[log10_ecorr])) * pmask for (log10_ecorr, pmask) in zip(log10_ecorrs, pmasks))
        getphi.params = params

        gp = utils.VariableGP(kernels.NoiseMatrix1D_var(getphi), Umatall)
        gp.index = {f'{psr.name}_{name}_coefficients({Umatall.shape[1]})': slice(0,Umatall.shape[1])} # better for cosine
        gp.name, gp.pos = psr.name, psr.pos
        gp.gpname, gp.gpcommon = name, []

        return gp


# timing model

def makegp_improper(psr, fmat, constant=1.0e40, name='improperGP', variable=False, project=False):
    if variable:
        phi = utils.jnparray(constant * np.ones(fmat.shape[1]))

        def getphi(params):
            return phi
        getphi.params = []

        gp = utils.VariableGP(kernels.NoiseMatrix1D_var(getphi), fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}
    else:
        gp = utils.ConstantGP(kernels.NoiseMatrix1D_novar(constant * np.ones(fmat.shape[1])), fmat)
        gp.index = {f'{psr.name}_{name}_coefficients({fmat.shape[1]})': slice(0, fmat.shape[1])}

    gp.name = psr.name
    gp.gpname = name
    # `project=True` marks this improper GP to be marginalized by orthogonal
    # projection (the exact flat-prior limit) rather than by feeding its huge
    # prior variance through the Woodbury solve -- the float32-safe path. See
    # See docs/design/single_precision/ (timing-model projection).
    # Off by default, so existing models are byte-identical.
    gp.project = project

    return gp

def makegp_timing(psr, constant=None, variance=None, svd=False, scale=1.0, variable=False, project=False):
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
            return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable, project=project)
        else:
            raise ValueError("signals.makegp_timing() can take a specification of _either_ `constant` or `variance`.")

    return makegp_improper(psr, fmat, constant=constant, name='timingmodel', variable=variable, project=project)

def makegp_standard_normal(psr, basis, name='standardnormalGP'):
    """Proper GP with an identity (unit-normal) coefficient prior on `basis`.

    Unlike makegp_improper (constant=1e40 improper/flat), this is a genuinely
    proper ConstantGP whose coefficient covariance is exactly ones(k): its
    coefficients c ~ Normal(0, I), and its log-determinant is retained. No
    projection and no column normalization -- the caller passes the unnormalized
    basis (e.g. a prior-normal z Jacobian) whose unit coefficient variance is the
    physical prior.
    """
    fmat = np.asarray(basis, dtype=np.float64)
    if fmat.ndim != 2:
        raise ValueError("makegp_standard_normal basis must be 2-D (n_toa, k)")
    k = fmat.shape[1]
    gp = utils.ConstantGP(kernels.NoiseMatrix1D_novar(np.ones(k)), fmat)
    gp.index = {f'{psr.name}_{name}_coefficients({k})': slice(0, k)}
    gp.name = psr.name
    gp.gpname = name
    gp.project = False
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

    fmat, fnorm = utils.jnparray(fmat), utils.jnparray(fref / psr.freqs)
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

    # f, df = utils.jnparray(f), utils.jnparray(df)
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

    gp = utils.VariableGP(kernels.NoiseMatrix12D_var(priorfunc), fmatfunc if callable(fmat) else fmat)
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
            vpars = [utils.jnparray([params[arg] for arg in argmap]) if isinstance(argmap, list) else params[argmap]
                    for argmap in argmaps]
            return vprior(f, df, *vpars)

        priorfunc.params = sorted(set(sum([argmap if isinstance(argmap, list) else [argmap] for argmap in argmaps], [])))
        priorfunc.type = getattr(prior, 'type', None)

    gp = utils.VariableGP(kernels.VectorNoiseMatrix12D_var(priorfunc), fmats)
    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(len(f)*i,len(f)*(i+1))
                for i, psr in enumerate(psrs)}
    gp.gpname, gp.gpcommon = name, common

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
            return utils.jnparray([means(f, df, *psrpar.values(), **{arg: params[argname] for arg, argname in margmap.items()})
                                    for psrpar, margmap in zip(psrpars, margmaps)])
        meanfunc.params = sorted(set.union(*[set(margmap.values()) for margmap in margmaps]))

        gp.means = meanfunc

    return gp


# these support leave-one-out PPC

def makegp_fourier_delay(psr, components, T=None, name='fourierGP'):
    argname = f'{psr.name}_{name}_mean({components*2})'

    _, _, fmat = fourierbasis(psr, components, T)
    Fmat = utils.jnparray(fmat)

    def delayfunc(params):
        return utils.jnp.dot(Fmat, params[argname])
    delayfunc.params = [argname]

    return delayfunc

def makegp_fourier_variance(psr, components, T=None, name='fourierGP', noisedict={}):
    argname = f'{psr.name}_{name}_variance({components*2},{components*2})'

    _, _, fmat = fourierbasis(psr, components, T)

    if argname in noisedict:
        return utils.ConstantGP(kernels.NoiseMatrix2D_novar(noisedict[argname]), fmat)
    else:
        def priorfunc(params):
            return params[argname]
        priorfunc.params = [argname]

        return utils.VariableGP(kernels.NoiseMatrix2D_var(priorfunc), fmat)

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
    f, df = utils.jnparray(fs[0]), utils.jnparray(dfs[0])

    def priorfunc(params):
        return jnp.concatenate([prior(f, df, *[params[arg] for arg in argmap]) for argmap in argmaps])
    priorfunc.params = sorted(set(sum(argmaps, [])))

    def invprior(params):
        p = priorfunc(params)
        return 1.0 / p, jnp.sum(jnp.log(p))
    invprior.params = priorfunc.params

    gp = utils.GlobalVariableGP(kernels.NoiseMatrix1D_var(priorfunc), fmats)
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
    f, df = utils.jnparray(fs[0]), utils.jnparray(dfs[0])

    orfmats = [utils.jnparray([[orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs]) for orf in orfs]

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
            invorf, orflogdet = utils.jnparray(np.linalg.inv(orfmat)), np.linalg.slogdet(orfmat)[1]
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

            orfcf = utils.jsp.linalg.cho_factor(orfmat)
            def factors(params):
                phi = prior(f, df, *[params[arg] for arg in argmap])
                phicf = utils.jsp.linalg.cho_factor(phi)

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
    # hack for metamath to properly
    # set phiinv
    nm =kernels.NoiseMatrix12D_var(priorfunc)
    nm.inv =invprior
    gp = utils.GlobalVariableGP(nm, fmats)
    gp.Phi_inv, gp.factors = invprior, factors

    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})':
                slice(len(f)*i, len(f)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]
    # introspection tags read by discovery.summary
    gp.gpname, gp.gpcommon = name, common
    gp.orfnames = [orf.__name__ for orf in orfs]

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


def CompoundGlobalGP(gplist):
    """Combine multiple GlobalVariableGPs (e.g. HD + monopole) into one.

    Backend-agnostic replacement for the legacy ``matrix.CompoundGlobalGP``:
    builds the combined block-structured prior through the ``_kernels`` factory
    and ``utils.GlobalVariableGP``, so it yields matrix.py classes in matrix
    mode and metamath classes in metamath mode. Reads only mode-neutral GP
    attributes (``gp.Phi.getN``, ``gp.Phi.getN.params``, ``gp.Phi_inv``) -- the
    same surface ``makeglobalgp_fourier`` populates in either mode.
    """
    if not all(isinstance(gp, utils.GlobalVariableGP) for gp in gplist):
        raise NotImplementedError("Cannot concatenate these types of GlobalGPs.")

    fmats = [np.hstack(F) for F in zip(*[gp.Fs for gp in gplist])]
    npsr = len(fmats)
    ngps = [gp.Fs[0].shape[1] for gp in gplist]
    allgp = sum(ngps)
    offsets = [0] + list(np.cumsum(ngps))[:-1]

    def _phi_params(gp):
        return list(getattr(gp.Phi.getN, 'params', []))

    def _is_2d(gp):
        return getattr(gp.Phi.getN, 'type', None) is jax.Array

    allparams = sorted(set().union(*[set(_phi_params(gp)) for gp in gplist])) if gplist else []

    if all(not _is_2d(gp) for gp in gplist):
        # all-diagonal global priors: interleave per-pulsar diagonal blocks
        def priorfunc(params):
            ret = jnp.zeros(npsr * allgp, 'd')
            for gp, ngp, offset in zip(gplist, ngps, offsets):
                phi = gp.Phi.getN(params)
                for i in range(npsr):
                    ret = ret.at[i*allgp+offset:i*allgp+offset+ngp].set(phi[i*ngp:(i+1)*ngp])
            return ret
        priorfunc.params = allparams

        multigp = utils.GlobalVariableGP(kernels.NoiseMatrix1D_var(priorfunc), fmats)
    else:
        # dense (cross-pulsar) global priors, e.g. HD: place each gp's
        # (npsr*ngp)^2 block matrix into the combined block-diagonal-by-gp layout.
        def priorfunc(params):
            ret = jnp.zeros((npsr*allgp, npsr*allgp), 'd')
            for gp, ngp, offset in zip(gplist, ngps, offsets):
                phi = gp.Phi.getN(params)
                if phi.ndim == 1:
                    phi = jnp.diag(phi)
                for i in range(npsr):
                    for j in range(npsr):
                        ret = ret.at[i*allgp+offset:i*allgp+offset+ngp,
                                     j*allgp+offset:j*allgp+offset+ngp].set(
                            phi[i*ngp:(i+1)*ngp, j*ngp:(j+1)*ngp])
            return ret
        priorfunc.params = allparams
        priorfunc.type = jax.Array

        phiinvs = [gp.Phi_inv for gp in gplist]
        if all(pi is not None for pi in phiinvs):
            def invprior(params):
                ret = jnp.zeros((npsr*allgp, npsr*allgp), 'd')
                ps, ls = zip(*[pi(params) for pi in phiinvs])
                for p, ngp, offset in zip(ps, ngps, offsets):
                    pinv = jnp.diag(p) if p.ndim == 1 else p
                    for i in range(npsr):
                        for j in range(npsr):
                            ret = ret.at[i*allgp+offset:i*allgp+offset+ngp,
                                         j*allgp+offset:j*allgp+offset+ngp].set(
                                pinv[i*ngp:(i+1)*ngp, j*ngp:(j+1)*ngp])
                return ret, sum(ls)
            invprior.params = allparams
            invprior.type = jax.Array
        else:
            invprior = None

        nm = kernels.NoiseMatrix2D_var(priorfunc)
        nm.inv = invprior
        multigp = utils.GlobalVariableGP(nm, fmats)
        multigp.Phi_inv = invprior

    index, cnt = {}, 0
    for vars in zip(*[gp.index.items() for gp in gplist]):
        for var, sli in vars:
            width = sli.stop - sli.start
            index[var] = slice(cnt, cnt + width)
            cnt = cnt + width
    multigp.index = index

    return multigp


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
        fs, zs = utils.jnparray(freqs[i_cutoff:]), jnp.zeros(i_cutoff)
    else:
        fs = utils.jnparray(freqs)

    def covmat(*args):
        if cutoff is not None:
            psd = jnp.concatenate([zs, psdfunc(fs, 1.0, *args[2:])])
        else:
            psd = psdfunc(fs, 1.0, *args[2:])

        fullpsd = jnp.concatenate((psd, psd[-2:0:-1]))
        Cfreq = jnp.fft.ifft(fullpsd, norm='backward')
        Ctau = Cfreq.real * len(fullpsd) * df / 2

        return utils.jsp.linalg.toeplitz(Ctau[:scaled_components:fmax_factor])
    covmat.__signature__ = inspect.signature(psdfunc)
    covmat.type = jax.Array

    return covmat

def makegp_fftcov(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, fourierbasis=None, common=[], name='fftcovGP'):
    T = getspan(psr) if T is None else T
    return makegp_fourier(psr, psd2cov(prior, components, T, oversample, fmax_factor, cutoff), components, T=T,
                          fourierbasis=(make_timeinterpbasis(start_time=t0, order=order) if fourierbasis is None else fourierbasis),
                          common=common, name=name)

def makegp_fftcov_dm(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='dm_gp', fref=1400.0):
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

def makegp_fftcov_chrom(psr, prior, components, T=None, t0=None, order=1, oversample=3, fmax_factor=1, cutoff=1, common=[], name='chrom_gp', fref=1400.0):
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


# log-space PSD constants (computed once at module load)
_LOG10_FYR  = _math.log10(const.fyr)
_LOG10_NORM = -_math.log10(12.0) - 2.0 * _math.log10(_math.pi)
_LN10       = _math.log(10.0)
_KAPPA      = 0.1  # fixed transition smoothness of the broken power-law model form


# single power laws

def make_powerlaw(*, gamma=None, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Power-law PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \frac{A^2}{12\pi^2}\, f_{\rm yr}^{\gamma-3}\, f^{-\gamma}\, \Delta f

    where :math:`A = 10^{\log_{10}A}`.  Evaluated in log-space; output
    clipped to :math:`[10^{\rm low\_clip},\, 10^{\rm high\_clip}]` s\ :sup:`2`.

    Parameters
    ----------
    gamma : float or None
        Fixed spectral index.  If None (default), ``gamma`` is a sampled
        parameter of the returned function.
    scale : float
        Multiplies :math:`\Phi` by ``scale**2``.
    low_clip : float
        Log10 floor of output in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling of output in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``powerlaw(f, df, log10_A[, gamma])``
    """
    _s2 = 2.0 * _math.log10(scale)

    if gamma is None:
        def powerlaw(f, df, log10_A, gamma):
            log10_phi = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                         - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))
    else:
        _g = float(gamma)
        _g_term = (_g - 3.0) * _LOG10_FYR
        def powerlaw(f, df, log10_A):
            log10_phi = (2.0 * log10_A + _g_term
                         - _g * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))

    return powerlaw


powerlaw = make_powerlaw()


# ---------------------------------------------------------------------------
# Pivot-amplitude power-law parameterization
# ---------------------------------------------------------------------------

import dataclasses as _dataclasses  # noqa: E402


@_dataclasses.dataclass(frozen=True)
class PowerLawParameterization:
    r"""Amplitude/slope parameterization for a power-law GP.

    The standard ``make_powerlaw`` samples ``log10_A`` at the fixed reference
    frequency ``f_ref = 1/yr``, where the amplitude and slope ``gamma`` are
    strongly correlated. Sampling the amplitude at a *pivot* frequency
    ``f_pivot`` near the data's sensitivity peak decorrelates them:

    .. math::

        \log_{10} A_{\rm ref} = \log_{10} A_{\rm pivot}
            + \tfrac12\, \gamma\, \log_{10}(f_{\rm pivot} / f_{\rm ref}).

    The map ``(log10_A_pivot, gamma) -> (log10_A_ref, gamma)`` is affine with unit
    Jacobian determinant, so it needs no density correction. ``amplitude_reference_frequency``
    is where the decoded/displayed ``log10_A`` is reported (``1/yr``, matching the
    PSD's internal reference). ``slope_pivot_frequency`` is either an explicit
    frequency in Hz or ``"sensitivity_weighted"`` (resolved once from the fixed
    reference-noise metric via :func:`sensitivity_weighted_pivot_frequency`).
    """

    amplitude_reference_frequency: float = const.fyr
    slope_pivot_frequency: "float | typing.Literal['sensitivity_weighted']" = (
        "sensitivity_weighted"
    )

    def resolve_pivot_frequency(self, *, freqs=None, weights=None) -> float:
        """Return the concrete pivot frequency in Hz.

        For ``"sensitivity_weighted"`` the per-frequency ``freqs`` (Hz) and their
        sensitivity ``weights`` are required; an explicit numeric pivot is
        returned as-is.
        """
        spf = self.slope_pivot_frequency
        if isinstance(spf, str):
            if spf != "sensitivity_weighted":
                raise ValueError(
                    f"slope_pivot_frequency must be a float or "
                    f"'sensitivity_weighted'; got {spf!r}")
            if freqs is None or weights is None:
                raise ValueError(
                    "the 'sensitivity_weighted' pivot needs freqs and weights "
                    "from the fixed reference-noise metric")
            return sensitivity_weighted_pivot_frequency(freqs, weights)
        return float(spf)


def sensitivity_weighted_pivot_frequency(freqs, weights) -> float:
    r"""Sensitivity-weighted geometric-mean pivot frequency.

    .. math:: \log f_{\rm pivot} = \frac{\sum_j w_j \log f_j}{\sum_j w_j}

    ``freqs`` (Hz) and ``weights`` are per-frequency (one entry per sine/cosine
    pair). Weights come from :func:`fourier_sensitivity_weights`.
    """
    f = np.asarray(freqs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if f.shape != w.shape or f.ndim != 1:
        raise ValueError("freqs and weights must be 1-D arrays of equal length")
    wsum = float(np.sum(w))
    if not wsum > 0.0:
        raise ValueError("sensitivity weights must sum to a positive value")
    return float(np.exp(np.sum(w * np.log(f)) / wsum))


def fourier_sensitivity_weights(fmat, reference_noise) -> np.ndarray:
    r"""Per-frequency sensitivity weights ``w_j = tr(F_j^T N0^-1 F_j)``.

    ``fmat`` is a discovery Fourier basis ``(n_toa, 2C)`` with sine/cosine pairs
    ordered ``[sin f1, cos f1, sin f2, cos f2, ...]``; ``F_j`` is columns
    ``[2j, 2j+1]``. ``reference_noise`` is a frozen reference-noise operator
    (``.solve(rhs) -> (N0^-1 rhs, logdet)``), so the weights use the exact frozen
    ``N0`` rather than live noise parameters.
    """
    F = np.asarray(fmat, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] % 2 != 0:
        raise ValueError("fmat must be (n_toa, 2C) with sine/cosine pairs")
    N0invF, _ = reference_noise.solve(F)
    per_col = np.einsum("ij,ij->j", F, np.asarray(N0invF, dtype=np.float64))
    return per_col[0::2] + per_col[1::2]


def reference_log10_amplitude(log10_A_pivot, gamma, *, f_pivot,
                              parameterization=None):
    """Convert a sampled ``log10_A_pivot`` to ``log10_A`` at the reference
    frequency; used to decode/display amplitudes at ``1/yr``."""
    if parameterization is None:
        parameterization = PowerLawParameterization()
    f_ref = float(parameterization.amplitude_reference_frequency)
    shift = 0.5 * _math.log10(float(f_pivot) / f_ref)
    return log10_A_pivot + gamma * shift


def make_powerlaw_pivot(*, f_pivot, parameterization=None, gamma=None,
                        scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Pivot-amplitude power-law PSD factory.

    Identical spectral form to :func:`make_powerlaw`, but the sampled amplitude
    ``log10_A_pivot`` is defined at ``f_pivot`` rather than the reference
    frequency. The returned function's amplitude argument is named
    ``log10_A_pivot`` (unambiguous public parameter name); decode the reference
    amplitude with :func:`reference_log10_amplitude`.

    Returns ``powerlaw(f, df, log10_A_pivot[, gamma])``.
    """
    if parameterization is None:
        parameterization = PowerLawParameterization()
    f_ref = float(parameterization.amplitude_reference_frequency)
    shift = 0.5 * _math.log10(float(f_pivot) / f_ref)  # log10_A_ref = A_pivot + gamma*shift
    _s2 = 2.0 * _math.log10(scale)

    if gamma is None:
        def powerlaw(f, df, log10_A_pivot, gamma):
            log10_A = log10_A_pivot + gamma * shift
            log10_phi = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                         - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))
    else:
        _g = float(gamma)
        _shift_g = _g * shift
        _g_term = (_g - 3.0) * _LOG10_FYR
        def powerlaw(f, df, log10_A_pivot):
            log10_A = log10_A_pivot + _shift_g
            log10_phi = (2.0 * log10_A + _g_term
                         - _g * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))

    return powerlaw


def make_brokenpowerlaw(*, gamma=None, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Broken power-law PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \frac{A^2}{12\pi^2}\, f_{\rm yr}^{\gamma-3}\, f^{-\gamma}
                  \left(1 + \left(\frac{f}{f_b}\right)^{1/\kappa}\right)^{\kappa\gamma}
                  \Delta f

    where :math:`f_b = 10^{\log_{10}f_b}` and :math:`\kappa = 0.1`.
    The broken factor is computed via ``logaddexp`` to avoid float32 overflow.
    Output clipped to :math:`[10^{\rm low\_clip},\, 10^{\rm high\_clip}]` s\ :sup:`2`.

    Parameters
    ----------
    gamma : float or None
        Fixed spectral index.  If None (default), ``gamma`` is sampled.
    scale : float
        Multiplies :math:`\Phi` by ``scale**2``.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``brokenpowerlaw(f, df, log10_A[, gamma], log10_fb)``
    """
    _s2 = 2.0 * _math.log10(scale)

    if gamma is None:
        def brokenpowerlaw(f, df, log10_A, gamma, log10_fb):
            z = (jnp.log(f) - log10_fb * _LN10) / _KAPPA
            log10_phi = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                         - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM
                         + _KAPPA * gamma * jnp.logaddexp(0.0, z) / _LN10 + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))
    else:
        _g = float(gamma)
        _g_term = (_g - 3.0) * _LOG10_FYR
        _kg = _KAPPA * _g
        def brokenpowerlaw(f, df, log10_A, log10_fb):
            z = (jnp.log(f) - log10_fb * _LN10) / _KAPPA
            log10_phi = (2.0 * log10_A + _g_term
                         - _g * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM
                         + _kg * jnp.logaddexp(0.0, z) / _LN10 + _s2)
            return utils.to_working(10.0 ** jnp.clip(log10_phi, low_clip, high_clip))

    return brokenpowerlaw


brokenpowerlaw = make_brokenpowerlaw()


def make_freespectrum(*, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Free-spectrum PSD factory.

    Returns a function evaluating

    .. math::

        \Phi_i = 10^{2\rho_i}

    repeated for sine/cosine pairs.  Output clipped to
    :math:`[10^{\rm low\_clip},\, 10^{\rm high\_clip}]` s\ :sup:`2`.

    Parameters
    ----------
    scale : float
        Multiplies :math:`\Phi` by ``scale**2``.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``freespectrum(f, df, log10_rho)``
    """
    _s2 = 2.0 * _math.log10(scale)

    def freespectrum(f, df, log10_rho: typing.Sequence):
        log10_phi = 2.0 * log10_rho + _s2
        return utils.to_working(jnp.repeat(10.0 ** jnp.clip(log10_phi, low_clip, high_clip), 2))

    return freespectrum


freespectrum = make_freespectrum()


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
        if utils.jnp == jnp:
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

def makepowerlaw_crn(components, crn_gamma='variable', *, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Combined IRN + CRN power-law PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \Phi_{\rm pl}(f;\, A, \gamma)
                  + \Phi_{\rm pl}(f_{1:2N_c};\, A_{\rm crn}, \gamma_{\rm crn})

    where :math:`N_c` = ``components`` and both terms use the standard
    power-law form.

    Parameters
    ----------
    components : int
        Number of CRN Fourier components; CRN is added to the first
        ``2 * components`` frequency bins.
    crn_gamma : float, 'variable', or None
        Fixed CRN spectral index.  ``'variable'`` or ``None`` makes it a
        sampled parameter of the returned function.
    scale : float
        Multiplies :math:`\Phi` by ``scale**2`` for both components.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``powerlaw_crn(f, df, log10_A, gamma, crn_log10_A[, crn_gamma])``
    """
    _s2 = 2.0 * _math.log10(scale)

    if utils.jnp == jnp:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            log10_phi = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                         - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
            phi = 10.0 ** jnp.clip(log10_phi, low_clip, high_clip)
            log10_crn = (2.0 * crn_log10_A + (crn_gamma - 3.0) * _LOG10_FYR
                         - crn_gamma * jnp.log10(f[:2*components])
                         + jnp.log10(df[:2*components]) + _LOG10_NORM + _s2)
            return utils.to_working(phi.at[:2*components].add(10.0 ** jnp.clip(log10_crn, low_clip, high_clip)))
    elif utils.jnp == np:
        def powerlaw_crn(f, df, log10_A, gamma, crn_log10_A, crn_gamma):
            phi = (10.0**(2.0 * log10_A)) / 12.0 / np.pi**2 * const.fyr ** (gamma - 3.0) * f ** (-gamma) * df
            phi[:2*components] += ((10.0**(2.0 * crn_log10_A)) / 12.0 / np.pi**2 *
                                   const.fyr ** (crn_gamma - 3.0) * f[:2*components] ** (-crn_gamma) * df[:2*components])
            return phi

    if crn_gamma not in ('variable', None):
        return utils.partial(powerlaw_crn, crn_gamma=crn_gamma)
    else:
        return powerlaw_crn


def make_powerlaw_brokencrn(*, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""IRN power-law + CRN broken power-law PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \Phi_{\rm pl}(f;\, A, \gamma)
                  + \Phi_{\rm bpl}(f;\, A_{\rm crn}, \gamma_{\rm crn}, f_{b,{\rm crn}})

    Clip applied per-component before summing.

    Parameters
    ----------
    scale : float
        Multiplies :math:`\Phi` by ``scale**2`` for both components.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``powerlaw_brokencrn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb)``
    """
    _s2 = 2.0 * _math.log10(scale)

    def powerlaw_brokencrn(f, df, log10_A, gamma, crn_log10_A, crn_gamma, crn_log10_fb):
        log10_irn = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                     - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM + _s2)
        z_crn = (jnp.log(f) - crn_log10_fb * _LN10) / _KAPPA
        log10_crn = (2.0 * crn_log10_A + (crn_gamma - 3.0) * _LOG10_FYR
                     - crn_gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM
                     + _KAPPA * crn_gamma * jnp.logaddexp(0.0, z_crn) / _LN10 + _s2)
        return utils.to_working(10.0 ** jnp.clip(log10_irn, low_clip, high_clip)
                                + 10.0 ** jnp.clip(log10_crn, low_clip, high_clip))

    return powerlaw_brokencrn


powerlaw_brokencrn = make_powerlaw_brokencrn()


def make_brokenpowerlaw_brokencrn(*, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""IRN broken power-law + CRN broken power-law PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \Phi_{\rm bpl}(f;\, A, \gamma, f_b)
                  + \Phi_{\rm bpl}(f;\, A_{\rm crn}, \gamma_{\rm crn}, f_{b,{\rm crn}})

    Clip applied per-component before summing.

    Parameters
    ----------
    scale : float
        Multiplies :math:`\Phi` by ``scale**2`` for both components.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``brokenpowerlaw_brokencrn(f, df, log10_A, gamma, log10_fb, crn_log10_A, crn_gamma, crn_log10_fb)``
    """
    _s2 = 2.0 * _math.log10(scale)

    def brokenpowerlaw_brokencrn(f, df, log10_A, gamma, log10_fb,
                                 crn_log10_A, crn_gamma, crn_log10_fb):
        z_irn = (jnp.log(f) - log10_fb * _LN10) / _KAPPA
        log10_irn = (2.0 * log10_A + (gamma - 3.0) * _LOG10_FYR
                     - gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM
                     + _KAPPA * gamma * jnp.logaddexp(0.0, z_irn) / _LN10 + _s2)
        z_crn = (jnp.log(f) - crn_log10_fb * _LN10) / _KAPPA
        log10_crn = (2.0 * crn_log10_A + (crn_gamma - 3.0) * _LOG10_FYR
                     - crn_gamma * jnp.log10(f) + jnp.log10(df) + _LOG10_NORM
                     + _KAPPA * crn_gamma * jnp.logaddexp(0.0, z_crn) / _LN10 + _s2)
        return utils.to_working(10.0 ** jnp.clip(log10_irn, low_clip, high_clip)
                                + 10.0 ** jnp.clip(log10_crn, low_clip, high_clip))

    return brokenpowerlaw_brokencrn


brokenpowerlaw_brokencrn = make_brokenpowerlaw_brokencrn()


def makefreespectrum_crn(components, *, scale=1.0, low_clip=-18.0, high_clip=-9.0):
    r"""Combined IRN + CRN free-spectrum PSD factory.

    Returns a function evaluating

    .. math::

        \Phi(f) = \Phi_{\rm fs}(f;\, \boldsymbol{\rho})
                  + \Phi_{\rm fs}(f_{1:2N_c};\, \boldsymbol{\rho}_{\rm crn})

    where :math:`N_c` = ``components``.

    Parameters
    ----------
    components : int
        Number of CRN components; CRN added to first ``2 * components`` bins.
    scale : float
        Multiplies :math:`\Phi` by ``scale**2`` for both components.
    low_clip : float
        Log10 floor in s\ :sup:`2` (default -18).
    high_clip : float
        Log10 ceiling in s\ :sup:`2` (default -9).

    Returns
    -------
    callable
        ``freespectrum_crn(f, df, log10_rho, crn_log10_rho)``
    """
    _s2 = 2.0 * _math.log10(scale)

    if utils.jnp == jnp:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = jnp.repeat(10.0 ** jnp.clip(2.0 * log10_rho + _s2, low_clip, high_clip), 2)
            crn = jnp.repeat(10.0 ** jnp.clip(2.0 * crn_log10_rho + _s2, low_clip, high_clip), 2)
            return utils.to_working(phi.at[:2*components].add(crn))
    elif utils.jnp == np:
        def freespectrum_crn(f, df, log10_rho: typing.Sequence, crn_log10_rho: typing.Sequence):
            phi = np.repeat(10.0**(2.0 * log10_rho), 2)
            phi[:2*components] += np.repeat(10.0**(2.0 * crn_log10_rho), 2)
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

    psrpars = {arg: utils.jnparray(getattr(psr, arg)) for arg in args if hasattr(psr, arg)}

    def delayfunc(params):
        return delay(**psrpars, **{arg: params[argname] for arg,argname in argmap.items()})
    delayfunc.params = sorted(argmap.values())
    delayfunc.name = name

    return delayfunc

# use with makedelay to set residuals dynamically from arrays
def getresiduals(y):
    return -y


def make_extsignal_fourier(psrs, coefffunc, components, T=None, common=[],
                           name='extsignal'):
    """Build a deterministic signal carried on its OWN Fourier basis.

    Returns a ``utils.ExtSignal`` for use as ``ArrayLikelihood(extsignals=[...])``.
    Unlike a GP it has no prior: its Fourier coefficients are a deterministic
    function of a few physical parameters (``coefffunc``). The likelihood folds
    it in via cross-terms with the GP basis -- see
    ``VectorWoodburyKernel_varP.make_kernelproduct_gpcomponent``.

    Parameters
    ----------
    psrs : list of `discover.Pulsar` objects
        Same order as the ArrayLikelihood's pulsar list.
    coefffunc : callable
        Per-pulsar map from physical parameters to a length-``2*components``
        Fourier-coefficient vector. Its first two positional arguments must be
        ``f, df`` (bound here to the basis); any argument that is a pulsar
        attribute (``pos``, ``mintoa``, ...) is bound from the pulsar; the rest
        become sampled parameters. Example: ``deterministic.makefourier_binary()``.
        If they are in `common` then they are common to the array. Otherwise one parameter
        per pulsar is created.
    components : int
        Number of frequency bins for this signal's basis.
    T : float, optional
        Total baseline time (in seconds) for the Fourier basis (default: per-pulsar span).
    common : list of str
        Parameter names shared across pulsars (e.g. CW earth-term parameters).
    name : str
        Parameter-name prefix and ExtSignal name.
    """
    fs, dfs, Fs = [], [], []
    for psr in psrs:
        f, df, fmat = fourierbasis(psr, components, T)
        fs.append(np.asarray(f))
        dfs.append(np.asarray(df))
        # keep Fs host-side: TOA-scale, only consumed by host-side trace-time
        # collapse in make_kernelproduct_gpcomponent.
        Fs.append(np.asarray(fmat))
    f_arr = utils.jnparray(np.stack(fs))      # (npsr, 2*components)
    df_arr = utils.jnparray(np.stack(dfs))

    # inspect coefffunc: arguments after the leading f, df
    argspec = inspect.getfullargspec(coefffunc)
    args = argspec.args + [a for a in argspec.kwonlyargs
                           if a not in (argspec.kwonlydefaults or {})]
    sig_args = [a for a in args if a not in ('f', 'df')]

    is_attr = {a: hasattr(psrs[0], a) for a in sig_args}
    is_common = {a: (not is_attr[a]) and (a in common or f'{name}_{a}' in common)
                 for a in sig_args}

    def pname(psr, a):
        if a in common:
            return a
        if f'{name}_{a}' in common:
            return f'{name}_{a}'
        return f'{psr.name}_{name}_{a}'

    attr_arr = {a: utils.jnparray(np.stack([np.asarray(getattr(p, a))
                                             for p in psrs]))
                for a in sig_args if is_attr[a]}

    in_axes = (0, 0) + tuple(None if is_common[a] else 0 for a in sig_args)
    vfunc = jax.vmap(coefffunc, in_axes=in_axes)

    params_list = sorted(set(
        [pname(psrs[0], a) for a in sig_args if is_common[a]] +
        [pname(p, a) for p in psrs for a in sig_args
         if not is_attr[a] and not is_common[a]]))

    def coeffs(params):
        callargs = [f_arr, df_arr]
        for a in sig_args:
            if is_attr[a]:
                callargs.append(attr_arr[a])
            elif is_common[a]:
                callargs.append(params[pname(psrs[0], a)])
            else:
                callargs.append(utils.jnparray(
                    [params[pname(p, a)] for p in psrs]))
        return vfunc(*callargs)                 # (npsr, 2*components)
    coeffs.params = params_list

    return utils.ExtSignal(Fs, coeffs, name=name)


# Measurement-noise constructors live in measurement_noise.py (collapsed form);
# re-exported here so `signals.makenoise_measurement` / `ds.makenoise_measurement`
# resolve. Import at module end to avoid a circular import (measurement_noise
# imports signals for `makegp_ecorr` / `selection_backend_flags`).
from .measurement_noise import makenoise_measurement, makenoise_measurement_simple  # noqa: E402
