"""Kernel-constructor factory shared by `signals.py` / `deterministic.py`.

`signals.py` builds the per-pulsar noise/GP kernels by name (e.g.
`NoiseMatrix1D_var`, `NoiseMatrixSM_novar`). Which *implementation* those names
resolve to depends on the active kernel subsystem:

  - ``matrix``   — the legacy closure classes in `matrix.py`.
  - ``metamath`` — the graph classes in `metamath.py`.

This module is that switch, in explicit form. Call sites do
``from . import _kernels as kernels`` and then ``kernels.NoiseMatrix1D_var(...)``;
it resolves the name per `set_mode(...)` (driven by `discovery.config(kernels=)`):

  - in ``metamath`` mode, names in ``_METAMATH`` resolve to their metamath
    class; everything else falls through to ``matrix.*``.
  - in ``matrix`` mode, every name falls through to ``matrix.*`` **at call
    time**. (Reading `matrix.*` live is deliberate: the parity test harness's
    ``mh_patched`` route monkeypatches `matrix.*`, and that patch flows
    through here transparently.)

This replaces the persistent `matrix.*` monkeypatch that
`discovery._kernel_switch` used to install for the production metamath path;
the monkeypatch now survives only as the test harness's `mh_patched` route.

Names intentionally NOT in ``_METAMATH`` (e.g. ``NoiseMatrix2D_novar``,
``VectorNoiseMatrix12D_var``) fall through to `matrix.*` even in metamath
mode — preserving today's behavior for constructors that have no metamath
port yet. Closing those gaps is further coverage work.
"""
from . import matrix
from . import metamath as mh


# Canonical matrix-name -> metamath-class map. `_kernel_switch._PATCHES`
# imports this so the test monkeypatch and this factory never drift.
_METAMATH = {
    # noise variants collapse to a single metamath NoiseMatrix; dispatch
    # happens at runtime via Sym.solve based on tensor ndim. The 1D/2D marker
    # subclasses preserve isinstance dispatch performed by likelihood.py.
    "NoiseMatrix1D_novar":     mh.NoiseMatrix1D,
    "NoiseMatrix1D_var":       mh.NoiseMatrix1D,
    "NoiseMatrix2D_novar":     mh.NoiseMatrix2D,
    "NoiseMatrix2D_var":       mh.NoiseMatrix2D,
    "NoiseMatrix12D_var":      mh.NoiseMatrix12D,
    # Vector noise has no separate metamath class -- metamath's NoiseMatrix
    # dispatches by ndim at runtime. The matrix.py `VectorNoiseMatrix12D_var`
    # dispatcher must map here too: previously the monkeypatch swapped the leaf
    # `VectorNoiseMatrix1D_var` *inside* the dispatcher; the factory intercepts
    # only the call site, so the dispatcher itself must resolve to metamath.
    "VectorNoiseMatrix1D_var":  mh.NoiseMatrix1D,
    "VectorNoiseMatrix12D_var": mh.NoiseMatrix12D,

    # ecorr Sherman-Morrison: dedicated indexed-SM graph
    "NoiseMatrixSM_novar":     mh.NoiseMatrixSM,
    "NoiseMatrixSM_var":       mh.NoiseMatrixSM,

    # Woodbury + compound
    "WoodburyKernel":            mh.WoodburyKernel,
    "VectorWoodburyKernel_varP": mh.VectorWoodburyKernel,
    "CompoundGP":                mh.CompoundGP,
    "VectorCompoundGP":          mh.CompoundGP,
    "CompoundDelay":             mh.CompoundDelay,
}


_mode = "matrix"


def set_mode(mode):
    """Select which kernel implementation `K.<Name>` resolves to.

    Driven by `discovery.config(kernels=...)`. Idempotent.
    """
    global _mode
    if mode not in ("matrix", "metamath"):
        raise ValueError(
            f"unknown kernels {mode!r}; expected 'matrix' or 'metamath'")
    _mode = mode


def get_mode():
    return _mode


def require_metamath(feature):
    """Raise unless the metamath kernel path is active.

    New entry points are metamath-only: the legacy path is frozen for deletion,
    and implementing new features twice is exactly the duplication this branch
    exists to remove.
    """
    if _mode != "metamath":
        raise NotImplementedError(
            f"{feature} requires the metamath kernel path; call "
            "discovery.config(kernels='metamath') before building the model.")


# ---------------------------------------------------------------------------
# Collapsed constructors
# ---------------------------------------------------------------------------
# metamath has a single NoiseMatrix per ndim that accepts either a concrete
# array (fixed) or a callable (variable); matrix.py splits these into `_novar`
# / `_var` classes. These entry points let call sites construct noise WITHOUT
# the novar/var dispatch: pass an array or a callable and the right backend
# class is chosen here. In matrix mode the choice is `callable(arg)`; in
# metamath mode the single class handles both. `measurement_noise.py` uses
# these so its logic no longer enumerates the variant classes.

def NoiseMatrix1D(N):
    """Diagonal white-noise kernel; `N` is an array (fixed) or callable (var)."""
    if _mode == "metamath":
        return mh.NoiseMatrix1D(N)
    return matrix.NoiseMatrix1D_var(N) if callable(N) else matrix.NoiseMatrix1D_novar(N)


def NoiseMatrixSM(N, F, P):
    """Sherman-Morrison white+ecorr kernel. `N` (and `P`) array or callable."""
    if _mode == "metamath":
        return mh.NoiseMatrixSM(N, F, P)
    return (matrix.NoiseMatrixSM_var if callable(N)
            else matrix.NoiseMatrixSM_novar)(N, F, P)


def __getattr__(name):
    if _mode == "metamath" and name in _METAMATH:
        return _METAMATH[name]
    return getattr(matrix, name)
