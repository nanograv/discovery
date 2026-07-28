"""Measurement (white) noise constructors — collapsed form.

These are the Role-A functions from `signals.py`: the only ones whose return
value *is* the per-pulsar measurement-noise kernel ``N`` (everything else in
`signals.py` builds ``N`` only as a GP prior). They are reproduced here in
*collapsed* form: instead of selecting a matrix.py variant class by name
(`NoiseMatrix1D_novar` vs `_var`, `NoiseMatrixSM_novar` vs `_var`), they call
the variant-agnostic factory entry points `_kernels.NoiseMatrix1D` /
`_kernels.NoiseMatrixSM`, passing an array (fixed) or a callable (variable).
The factory picks the concrete class.

This is the strangler replacement for `signals.makenoise_measurement{,_simple}`.
While both coexist, the parity suite compares them in matrix mode (collapsed vs
original both build matrix.py classes, isolating that the collapse is faithful)
and in metamath mode (collapsed builds metamath kernels, compared to the matrix
oracle). Once green, `signals.py` is repointed here and the originals deleted.
"""
import numpy as np
import jax.numpy as jnp

from . import utils
from . import _kernels as kernels
from . import signals


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
        N = noise
    else:
        toaerrs = utils.jnparray(psr.toaerrs)

        def getnoise(params, tnequad=tnequad, add_equad=add_equad):
            if tnequad and add_equad:
                return params[efac]**2 * toaerrs**2 + 10.0**(2.0 * params[log10_tnequad])
            elif add_equad:
                return params[efac]**2 * (toaerrs**2 + 10.0**(2.0 * params[log10_t2equad]))
            else:
                return params[efac]**2 * toaerrs**2

        getnoise.params = params
        N = getnoise

    kern = kernels.NoiseMatrix1D(N)
    # introspection tag read by discovery.summary (harmless to the math path):
    # which white-noise parameters this kernel carries, and whether they were
    # fixed from the noise dictionary rather than left free.
    kern.measurement = {'name': 'measurement', 'params': params, 'psrname': psr.name,
                        'fixed': all(par in noisedict for par in params),
                        'ecorr': False}
    return kern


def makenoise_measurement(psr, noisedict={}, scale=1.0, tnequad=False, ecorr=False,
                          selection=None, vectorize=True,
                          outliers=False, enterprise=False):
    # default resolved lazily to avoid a def-time reference into `signals`
    # (which imports this module): signals re-exports these functions.
    if selection is None:
        selection = signals.selection_backend_flags
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

    is_const = all(par in noisedict for par in params)

    if is_const:
        if outliers:
            raise ValueError("No outlier scaling if white noise is fixed.")
        if tnequad:
            noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))
        N = noise
    else:
        if vectorize:
            toaerrs2, masks = utils.jnparray(scale**2 * psr.toaerrs**2), utils.jnparray([mask for mask in masks])

            if tnequad:
                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = utils.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = utils.jnparray([10.0**(2 * (logscale + params[log10_tnequad])) for log10_tnequad in log10_tnequads])

                    return (masks * (efac2[:,jnp.newaxis] * (alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
            else:

                def getnoise(params):
                    if outliers:
                        alpha_scaling = params[toaerr_scaling]
                    else:
                        alpha_scaling = 1.0
                    efac2  = utils.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = utils.jnparray([10.0**(2 * (logscale + params[log10_t2equad])) for log10_t2equad in log10_t2equads])

                    return (masks * efac2[:,jnp.newaxis] * ((alpha_scaling*toaerrs2)[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
        else:
            toaerrs, masks = utils.jnparray(scale * psr.toaerrs), [utils.jnparray(mask) for mask in masks]
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
        N = getnoise

    if ecorr:
        # ecorr folded into N via Sherman-Morrison. The ecorr GP supplies the
        # low-rank basis F and its prior P; const vs variable tracks `is_const`
        # exactly as the original did (P is the array `.N` or the callable `.getN`).
        egp = signals.makegp_ecorr(psr, noisedict=(noisedict if is_const else {}),
                                   enterprise=enterprise, scale=scale, selection=selection)
        P = egp.Phi.N if is_const else egp.Phi.getN
        kern = kernels.NoiseMatrixSM(N, egp.F, P)
        ecorr_backends = [bf for bf in sorted(set(selection(psr))) if bf != '']
        kern.measurement = {'name': 'measurement', 'params': params, 'psrname': psr.name,
                            'fixed': is_const, 'ecorr': True,
                            'ecorr_params': [f'{psr.name}_{b}_log10_ecorr' for b in ecorr_backends],
                            'ecorr_basis_shape': tuple(np.shape(egp.F))}
        return kern
    else:
        kern = kernels.NoiseMatrix1D(N)
        kern.measurement = {'name': 'measurement', 'params': params, 'psrname': psr.name,
                            'fixed': is_const, 'ecorr': False}
        return kern
