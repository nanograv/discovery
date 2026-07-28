"""Matrix.py → metamath patch mappings, shared by tests and ds.config.

Until `signals.py` is migrated to construct metamath kernel objects
directly, the metamath path needs `signals.py`'s `matrix.X(...)` calls to
resolve to the corresponding `metamath` classes. That's what this module
provides:

- ``_PATCHES`` — the matrix.* → metamath swap table.
- ``apply_patches()`` / ``restore_patches()`` — install / undo the swaps
  persistently. Used by ``ds.config(kernels=...)``.
- ``patched_kernels()`` — a context manager that installs the patches
  for the duration of a block (used by the test harness).

Once ``signals.py`` is migrated, every patch entry becomes a no-op (the
target name already produces a metamath kernel) and this module can be
deleted.
"""
from contextlib import contextmanager

from . import matrix
# Single source of truth for the matrix-name -> metamath-class map lives in
# `_kernels`; the test harness's `mh_patched` route reuses it here so the
# monkeypatch and the production factory can never drift apart.
from ._kernels import _METAMATH as _PATCHES


# `_originals` is None when no patches are installed; otherwise it is the
# saved dict of original attribute values to restore on `restore_patches()`.
_originals = None


def apply_patches():
    """Install the matrix→metamath patches on `discovery.matrix`. Idempotent.

    Subsequent calls while patches are already active are no-ops (the
    saved originals are not overwritten).
    """
    global _originals
    if _originals is not None:
        return
    saved = {}
    for name, replacement in _PATCHES.items():
        saved[name] = getattr(matrix, name)
        setattr(matrix, name, replacement)
    _originals = saved


def restore_patches():
    """Reverse `apply_patches()`. No-op if patches aren't installed."""
    global _originals
    if _originals is None:
        return
    for name, original in _originals.items():
        setattr(matrix, name, original)
    _originals = None


@contextmanager
def patched_kernels():
    """Context-manager form: install the patches for the duration of a block.

    Used by the parity test harness's `mh_patched` route. Composes correctly
    with persistent installation by `ds.config(kernels='metamath')`: on enter
    it captures whatever state is currently active and restores it on exit.
    """
    saved = {}
    for name, replacement in _PATCHES.items():
        saved[name] = getattr(matrix, name)
        setattr(matrix, name, replacement)
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(matrix, name, original)
