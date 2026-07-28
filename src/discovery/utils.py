"""Shared substrate for both kernel paths (`matrix.py` and `metamath.py`).

This module owns two things that both the legacy closure-based path and the
graph-based metamath path need, so that neither path has to depend on the
other for them:

1. **Numerical backend configuration.** `config(backend=..., factor=...)`
   populates module-level names here (`jnp`, `jsp`, `jnparray`, `jnpsplit`,
   `jnpnormal`, `matrix_factor`, `matrix_solve`, `matrix_norm`,
   `regularize_FtNmF`, ...) so downstream code can switch numpy vs jax,
   single vs double precision, and cholesky vs LU factorization through one
   entry point. `config()` runs once at import. Downstream code reads these
   as `utils.jnp`, `utils.jnparray`, etc.

   This previously lived in `matrix.py`; it moved here so the metamath path
   and the shared kernel primitives below no longer reach back into
   `matrix.py` for the backend.

2. **Shared kernel primitives.** Marker base classes (`Kernel`, `GP`,
   `ConstantGP`, ...) used for `isinstance` dispatch, the `ExtSignal`
   container, and the leaf indexed Sherman-Morrison helpers. Small, pure,
   no const/var bookkeeping; reused by both paths.
"""
from collections.abc import Sequence
import functools
import inspect

import numpy as np
import scipy as sp

import jax
import jax.numpy
import jax.scipy
import jax.tree_util


# ---------------------------------------------------------------------------
# Numerical backend configuration
# ---------------------------------------------------------------------------

def config(**kwargs):
    global jnp, jsp, jnparray, jnpzeros, intarray, jnpkey, jnpsplit, jnpnormal
    global matrix_factor, matrix_solve, matrix_norm, partial, SM_algorithm, regularize_FtNmF
    global single_precision, _working_dtype

    np.logdet = lambda a: np.sum(np.log(np.abs(a)))
    jax.numpy.logdet = lambda a: jax.numpy.sum(jax.numpy.log(jax.numpy.abs(a)))

    np.make2d = lambda a: a if a.ndim == 2 else np.diag(a)
    jax.numpy.make2d = lambda a: a if a.ndim == 2 else jax.numpy.diag(a)

    np.makearray = lambda a: np.array(a) if hasattr(a, '__len__') else a
    jax.numpy.makearray = lambda a: jax.numpy.array(a) if hasattr(a, '__len__') else a

    backend = kwargs.get('backend')

    if backend == 'numpy':
        jnp, jsp = np, sp

        jnparray = lambda a: np.array(a, dtype=np.float64)
        jnpzeros = lambda a: np.zeros(a, dtype=np.float64)
        intarray = lambda a: np.array(a, dtype=np.int64)

        jnpkey    = lambda seed: np.random.default_rng(seed)
        jnpsplit  = lambda gen: (gen, gen)
        jnpnormal = lambda gen, shape: gen.normal(size=shape)
        _working_dtype = np.float64
        single_precision = False
        partial = functools.partial
    elif backend == 'jax':
        jnp, jsp = jax.numpy, jax.scipy

        _wd = kwargs.get('working', jax.numpy.float64)
        if _wd == jax.numpy.float32 and not jax.config.x64_enabled:
            raise AssertionError(
                "config(working=float32) requires x64 enabled; "
                "call jax.config.update('jax_enable_x64', True) first."
            )
        _working_dtype = _wd
        single_precision = (_working_dtype == jax.numpy.float32)

        jnparray = lambda a: jnp.array(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        jnpzeros = lambda a: jnp.zeros(a, dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32)
        intarray = lambda a: jnp.array(a, dtype=jnp.int64)

        jnpkey    = lambda seed: jax.random.PRNGKey(seed)
        jnpsplit  = jax.random.split
        jnpnormal = jax.random.normal

        partial = jax.tree_util.Partial

    regularize_FtNmF = kwargs.get('regularize_FtNmF', single_precision)
    factor = kwargs.get('factor')

    if factor == 'cholesky':
        matrix_factor = jsp.linalg.cho_factor
        matrix_solve  = jsp.linalg.cho_solve
        matrix_norm   = 2.0
    elif factor == 'lu':
        matrix_factor = jsp.linalg.lu_factor
        matrix_solve  = jsp.linalg.lu_solve
        matrix_norm   = 1.0

    SM_algorithm = 'indexed'

config(backend='jax', factor='cholesky')

def rngkey(seed):
    return jnpkey(seed)

def working_dtype():
    return _working_dtype

def to_working(a):
    return jnp.asarray(a, dtype=_working_dtype)

# CG solver and Lanczos-Hutchinson logdet estimator, need matfree and jaxopt
try:
    import jaxopt
    from matfree import decomp, funm, stochtrace

    cgsolve = jaxopt.linear_solve.solve_cg

    def dense_funm_sym_eigh(matfun, clip=None):
        def fun(dense_matrix):
            eigvals, eigvecs = funm.linalg.eigh(dense_matrix)
            # optional clipping
            if clip:
                eigvals = jnp.clip(eigvals, a_min=1e-6)
            fx_eigvals = funm.func.vmap(matfun)(eigvals)
            return eigvecs @ funm.linalg.diagonal(fx_eigvals) @ eigvecs.T

        return fun

    def integrand_funm_sym_logdet(tridiag_sym, clip=None):
        dense_funm = dense_funm_sym_eigh(jnp.log, clip=clip)
        return funm.integrand_funm_sym(dense_funm, tridiag_sym)

    def make_logdet_estimator(ndim, num_matvecs=40, samples=1000, clip=None):
        #
        # log det A = tr log A = (1/S) \sum_i^S z_i^T (log A) z_i

        # set up Lanczos tridiagonalization interface, uses num_matvecs applications of A
        tridiag_sym = decomp.tridiag_sym(num_matvecs)

        # set up integrand for Lanczos quadrature
        problem = integrand_funm_sym_logdet(tridiag_sym, clip=clip)

        # generate `samples` random probes of shape
        sampler = stochtrace.sampler_normal(jnpzeros(ndim), num=samples)

        # combine problem and sampler into Hutchinson trace estimator
        estimator = stochtrace.estimator(problem, sampler=sampler)

        return estimator

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Shared kernel primitives (formerly kernel_helpers.py)
# ---------------------------------------------------------------------------

class Kernel:
    """Marker base class for kernel / GP / noise-matrix objects."""

    def __repr__(self):
        # white-noise kernels are tagged with `.measurement` by the
        # measurement-noise factories; show that nicely, else a plain label.
        meas = getattr(self, 'measurement', None)
        if meas is not None:
            kind = 'white noise' + (' + ECORR (Sherman-Morrison)' if meas.get('ecorr') else '')
            n = len(meas.get('params', [])) + len(meas.get('ecorr_params', []))
            state = 'fixed' if meas.get('fixed') else 'free'
            return f'<{kind} | {n} {state} params>'
        return f'<{type(self).__name__}>'


class ConstantKernel(Kernel):
    pass


class VariableKernel(Kernel):
    pass


class ConstantMatrix:
    pass


class VariableMatrix:
    pass


class NoiseMatrix:
    """Base type for noise-kernel objects. Subclassed by `matrix.NoiseMatrix*`
    and `metamath.NoiseMatrix`; carried as a marker for isinstance dispatch."""
    pass


class GP:
    pass


def _gp_params(Phi):
    """Free-parameter names of a prior kernel, across both backends: matrix.py
    exposes ``Phi.params``; metamath keeps the parameterized callable at
    ``Phi.getN.params``. Returns a (possibly empty) list."""
    pars = getattr(Phi, 'params', None)
    if pars:
        return list(pars)
    return list(getattr(getattr(Phi, 'getN', None), 'params', None) or [])


def _gp_repr(gp, cls):
    """Compact, defensive ``__repr__`` shared by the GP marker classes.

    Reads only attributes the factories already set, so an under-populated GP
    still reprs without raising. Shows the signal name, basis shape, ORF (for
    correlated GPs), and either the free parameters or a 'fixed prior' note.
    """
    bits = [cls]
    name = getattr(gp, 'gpname', None)
    if name:
        bits[0] = f"{cls} '{name}'"
    orfs = getattr(gp, 'orfnames', None)
    if orfs:
        bits[0] += f" ({', '.join(orfs)})"

    F = getattr(gp, 'F', None)
    Fs = getattr(gp, 'Fs', None)
    # a per-pulsar (common) GP keeps its bases as a list in either .F or .Fs
    if isinstance(F, (list, tuple)):
        Fs, F = F, None
    if F is not None and hasattr(F, 'shape'):
        bits.append('basis ' + '×'.join(str(d) for d in F.shape))
    elif Fs:
        npsr = len(Fs)
        nb = Fs[0].shape[1] if hasattr(Fs[0], 'shape') and Fs[0].ndim == 2 else '?'
        bits.append(f'{npsr} psr × {nb}')

    pars = _gp_params(getattr(gp, 'Phi', None))
    if pars:
        shown = ', '.join(pars[:4]) + (', …' if len(pars) > 4 else '')
        bits.append(f'params: {shown}')
    else:
        bits.append('fixed prior')

    return '<' + ' | '.join(bits) + '>'


class ConstantGP:
    """A Gaussian-process signal whose prior precision is fixed at trace time
    (no parameters). `Phi` is the prior covariance; `F` is the per-pulsar
    design matrix."""
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

    def __repr__(self):
        return _gp_repr(self, 'ConstantGP')


class VariableGP:
    """A Gaussian-process signal whose prior precision depends on parameters."""
    def __init__(self, Phi, F):
        self.Phi, self.F = Phi, F

    def __repr__(self):
        return _gp_repr(self, 'VariableGP')


class GlobalVariableGP:
    """Like VariableGP, but with per-pulsar design matrices in a list `Fs`.
    Factories returning a GlobalVariableGP should set `.index` as a dict
    of component-vector-name -> slice within the stacked Fs."""
    def __init__(self, Phi, Fs):
        self.Phi, self.Fs = Phi, Fs
        self.Phi_inv = None

    def __repr__(self):
        return _gp_repr(self, 'GlobalVariableGP')


class ExtSignal:
    """A deterministic signal carried on its own (non-GP) Fourier-style basis.

    Used for signals (e.g. a continuous wave) whose basis must extend to
    higher frequencies than the GP bases reach. Unlike a GP it has no prior:
    its coefficients are a deterministic function of a handful of physical
    parameters, supplied by ``coeffs``.

    The object is purely declarative and noise-free; all noise-dependent
    linear algebra (cross-terms with the GP basis and the data) lives in
    the kernelproduct_gpcomponent path of whichever backend is in use.

    Parameters
    ----------
    Fs : list of array
        Per-pulsar design matrices, one ``(ntoa_i, n_ext)`` array per pulsar,
        in the same pulsar order as the likelihood.
    coeffs : callable
        ``coeffs(params) -> (npsr, n_ext)`` deterministic coefficient map,
        with a ``.params`` attribute listing its parameter names.
    name : str
        Identifier for the signal.
    """
    def __init__(self, Fs, coeffs, name='extsignal'):
        self.Fs, self.coeffs, self.name = Fs, coeffs, name

    @property
    def params(self):
        return self.coeffs.params

    def __repr__(self):
        bits = [f"ExtSignal '{self.name}'"]
        if self.Fs and hasattr(self.Fs[0], 'shape') and self.Fs[0].ndim == 2:
            bits.append(f'{len(self.Fs)} psr × {self.Fs[0].shape[1]}')
        pars = list(self.params or [])
        if pars:
            bits.append('params: ' + ', '.join(pars[:4]) + (', …' if len(pars) > 4 else ''))
        return '<' + ' | '.join(bits) + '>'


def make_uind(U):
    """Pack 0/1 exposure matrix U (n_toa, n_epoch) into the index encoding
    expected by the indexed Sherman-Morrison primitives.

    Returns Uind of shape (n_epoch, max_per_epoch + 1) with TOA indices
    shifted by +1 so that 0 acts as a sentinel (used together with a
    zero-prepended y / +inf-prepended N).
    """
    U = np.asarray(U)

    # No epochs (e.g. an ECORR selection that matches no TOAs): return an
    # empty index table instead of taking the max of an empty array.
    if U.shape[1] == 0:
        return np.zeros((0, 1), 'i')

    maxcount = int(np.max(np.sum(U, axis=0)))
    Uind = np.zeros((U.shape[1], maxcount + 1), 'i')

    for i in range(U.shape[1]):
        ind = np.where(U[:, i])[0]
        Uind[i, 0:len(ind)] = ind + 1

    return Uind


def smup_ind(A, l, Amb, ind):
    Amu = 1.0 / A[ind]

    vtAmb = l * jnp.sum(Amb[ind])
    vtAmu = l * jnp.sum(Amu)

    return Amu * (vtAmb / (1.0 + vtAmu))


def smdp_ind(A, l, ind):
    Amu = 1.0 / A[ind]
    vtAmu = l * jnp.sum(Amu)

    return jnp.log1p(vtAmu)


vsmup_ind = jax.vmap(smup_ind, in_axes=(None, 0, None, 0))
vsmdp_ind = jax.vmap(smdp_ind, in_axes=(None, 0, 0))


def smup_ind_correct(yp, Np, Uind, P):
    corrections = vsmup_ind(Np, P, yp / Np, Uind)
    return (yp / Np).at[Uind.reshape(-1)].add(-corrections.reshape(-1))[1:]


vsmup_ind_correct = jax.vmap(smup_ind_correct, in_axes=(0, None, None, None))


# --- Sherman-Morrison whitening (apply K^{-1/2}, parallel to the solve above) ---
# These give W x with W = K^{-1/2} for K = diag(N) + F_ec diag(P) F_ec^T (ECORR).
# The ECORR indicators are orthogonal (one TOA per epoch), so K decouples per
# epoch and K^{-1/2} has the closed form with per-epoch factor alpha_k below.
# Used by the timing-model projection (float32-safe); see
# See docs/design/single_precision/ (timing-model projection).

def smwhiten_ind(A, l, xdivA, ind):
    """Per-epoch whitening correction: the rank-1 part of (K^{-1/2} x) for one epoch."""
    inv_d = 1.0 / A[ind]
    inv_sqrt_d = 1.0 / jnp.sqrt(A[ind])

    vtAmu = l * jnp.sum(inv_d)        # u_k = P_k sum_{i in k} 1/N_i
    vtAmb = l * jnp.sum(xdivA[ind])   # P_k sum_{i in k} x_i/N_i

    alpha = jnp.where(vtAmu > 1e-14,
                      (1.0 / jnp.sqrt(1.0 + vtAmu) - 1.0) / vtAmu,
                      -0.5)
    return inv_sqrt_d * (alpha * vtAmb)


vsmwhiten_ind = jax.vmap(smwhiten_ind, in_axes=(None, 0, None, 0))


def smwhiten_ind_correct(xp, Np, Uind, P):
    """W x for a single vector. xp/Np are sentinel-padded (index 0); drop it with [1:]."""
    corrections = vsmwhiten_ind(Np, P, xp / Np, Uind)
    return (xp / jnp.sqrt(Np)).at[Uind.reshape(-1)].add(corrections.reshape(-1))[1:]


vsmwhiten_ind_correct = jax.vmap(smwhiten_ind_correct, in_axes=(0, None, None, None))
