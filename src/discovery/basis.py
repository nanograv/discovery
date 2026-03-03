"""Lazy design matrix for Fourier-domain GPs.

Stores frequency metadata eagerly (small) and a recipe for
materializing the full (n_toa, 2*components) design matrix on demand.
This avoids storing N_psr copies simultaneously during construction.

Works with metamatrix: treated as a FuncLeaf in the graph,
materialized during the forward pass rather than at graph construction.
"""

import inspect
import numpy as np
import scipy.interpolate as si

from . import matrix


class Basis:
    """Lazy, optionally cached, design matrix.

    Parameters
    ----------
    f, df : array
        Frequency and bin-width arrays (small, stored eagerly).
    recipe : callable
        No params: recipe() -> fmat  (constant basis)
        With params: recipe(alpha=...) -> fmat  (variable basis)
    params : list of str, optional
        Parameter names required by recipe. Empty = constant basis.
    """

    __slots__ = ('f', 'df', '_recipe', 'params', '_cache')

    def __init__(self, f, df, recipe, params=None):
        self.f = f
        self.df = df
        self._recipe = recipe
        self.params = params or []
        self._cache = None

    @property
    def is_constant(self):
        return len(self.params) == 0

    def materialize(self, params={}):
        """Build the full design matrix, caching if constant."""
        if self.is_constant:
            if self._cache is None:
                self._cache = self._recipe()
            return self._cache
        return self._recipe(**{k: params[k] for k in self.params})

    def clear(self):
        """Free cached design matrix after compact products are built."""
        self._cache = None

    # metamatrix compat: callable with (params={}) signature
    def __call__(self, params={}):
        return self.materialize(params)

    # --- Factory methods ---

    @classmethod
    def fourier(cls, psr, components, T=None):
        """Standard Fourier basis (sin/cos pairs)."""
        from . import signals
        if T is None:
            T = signals.getspan(psr)

        f = np.arange(1, components + 1, dtype=np.float64) / T
        df = np.diff(np.concatenate((np.array([0]), f)))

        toas = psr.toas  # reference, not copy

        def recipe():
            fmat = np.zeros((toas.shape[0], 2 * components), dtype=np.float64)
            for i in range(components):
                fmat[:, 2*i]   = np.sin(2.0 * np.pi * f[i] * toas)
                fmat[:, 2*i+1] = np.cos(2.0 * np.pi * f[i] * toas)
            return fmat

        return cls(np.repeat(f, 2), np.repeat(df, 2), recipe)

    @classmethod
    def dmfourier(cls, psr, components, T=None, fref=1400.0):
        """DM Fourier basis (chromatic scaling with fixed alpha=2)."""
        base = cls.fourier(psr, components, T)
        Dm = (fref / psr.freqs) ** 2

        base_recipe = base._recipe
        def recipe():
            return base_recipe() * Dm[:, None]

        return cls(base.f, base.df, recipe)

    @classmethod
    def dmfourier_alpha(cls, psr, components, T=None, fref=1400.0):
        """DM Fourier basis with variable chromatic index alpha."""
        base = cls.fourier(psr, components, T)
        fnorm = fref / psr.freqs

        base_recipe = base._recipe
        def recipe(alpha):
            return matrix.jnparray(base_recipe()) * matrix.jnparray(fnorm)[:, None] ** alpha

        return cls(base.f, base.df, recipe, params=['alpha'])

    @classmethod
    def dmfourier_solar(cls, psr, components, T=None):
        """DM Fourier basis with solar wind shaping."""
        from . import solar
        base = cls.fourier(psr, components, T)
        shape = solar.make_solardm(psr)(1.0)

        base_recipe = base._recipe
        def recipe():
            return base_recipe() * shape[:, None]

        return cls(base.f, base.df, recipe)

    @classmethod
    def dmfourier_custom(cls, psr, components, alpha=2.0, tndm=False, T=None, fref=1400.0):
        """DM Fourier basis with fixed custom alpha and optional TNDM scaling."""
        base = cls.fourier(psr, components, T)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        base_recipe = base._recipe
        def recipe():
            return base_recipe() * Dm[:, None]

        return cls(base.f, base.df, recipe)

    @classmethod
    def timeinterp(cls, psr, components, T=None, start_time=None, order=1):
        """Time-interpolated covariance basis (linear or higher-order)."""
        from . import signals
        t0 = start_time if start_time is not None else np.min(psr.toas)
        if t0 > np.min(psr.toas):
            raise ValueError('Coarse time basis start must be earlier than earliest TOA.')
        if T is None:
            T = signals.getspan(psr)

        t_coarse = np.linspace(t0, t0 + T, components)
        dt_coarse = t_coarse[1] - t_coarse[0]

        toas = psr.toas

        if order == 1:
            # Fast linear interpolation without scipy
            def recipe():
                idx = np.arange(len(toas))
                idy = np.searchsorted(t_coarse, toas)
                idy[idy == 0] = 1
                Bmat = np.zeros((len(toas), len(t_coarse)), 'd')
                Bmat[idx, idy] = (toas - t_coarse[idy - 1]) / dt_coarse
                Bmat[idx, idy - 1] = (t_coarse[idy] - toas) / dt_coarse
                return Bmat
        else:
            def recipe():
                return si.interp1d(t_coarse, np.identity(components),
                                   kind=order)(toas).T

        return cls(t_coarse, dt_coarse, recipe)

    @classmethod
    def dmtimeinterp(cls, psr, components, T=None, start_time=None,
                     order=1, alpha=2.0, tndm=False, fref=1400.0):
        """DM-scaled time-interpolated basis."""
        base = cls.timeinterp(psr, components, T, start_time, order)

        if tndm:
            Dm = (fref / psr.freqs) ** alpha * np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
        else:
            Dm = (fref / psr.freqs) ** alpha

        base_recipe = base._recipe
        def recipe():
            return base_recipe() * Dm[:, None]

        return cls(base.f, base.df, recipe)

    @classmethod
    def from_existing(cls, psr, components, T=None, basis_func=None, **kwargs):
        """Wrap any existing (f, df, fmat_or_func) basis function.

        For truly lazy constant bases, use the classmethod factories above.
        This wrapper calls the basis function immediately to get f and df,
        but defers fmat computation for variable bases.
        """
        f, df, fmat_or_func = basis_func(psr, components, T, **kwargs)

        if callable(fmat_or_func):
            spec = inspect.getfullargspec(fmat_or_func)
            params = [a for a in spec.args if a not in ('f', 'df')]
            return cls(f, df, fmat_or_func, params=params)
        else:
            # Already materialized — wrap in closure for uniform interface
            fmat = fmat_or_func
            return cls(f, df, lambda: fmat)

    def __iter__(self):
        """Unpack as (f, df, self) — drop-in for legacy (f, df, fmat) returns."""
        return iter((self.f, self.df, self))
