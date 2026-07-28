"""Discovery"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from .const import *
from .matrix import *
from .params import *
from .prior import *
from .signals import *
from .likelihood import *
from .params import *
from .optimal import *
from .solar import *
from .pulsar import *
from .deterministic import *


_KERNELS = "metamath"
_LIKELIHOOD_CLASSES = ("PulsarLikelihood", "GlobalLikelihood", "ArrayLikelihood")


def config(kernels=None):
    """Select the kernel-implementation subsystem the top-level likelihoods use.

    Parameters
    ----------
    kernels : {'matrix', 'metamath'}, optional
        - 'matrix'  : the legacy closure-based path in `likelihood.py`,
                      built on `matrix.py` classes.
        - 'metamath': the graph-based path in `likelihood_metamath.py`,
                      built on `metamath.py` classes.

        Distinct from `utils.config(backend=...)`, which switches the
        underlying numerical backend (numpy vs jax). This switch picks
        which kernel implementation backs the likelihoods: it sets the
        `_kernels` factory mode (so `signals.py` builds the right kernels)
        and rebinds the top-level likelihood classes.

        When called, rebinds `discovery.PulsarLikelihood`,
        `discovery.GlobalLikelihood`, and `discovery.ArrayLikelihood` to the
        corresponding classes from the chosen module. Existing class
        references already imported into user code are NOT updated — call
        `config()` before constructing models.

    Returns the current kernels setting if called with no arguments.
    """
    global _KERNELS

    if kernels is None:
        return _KERNELS

    if kernels not in ("matrix", "metamath"):
        raise ValueError(
            f"unknown kernels {kernels!r}; expected 'matrix' or 'metamath'"
        )

    from . import _kernels

    _kernels.set_mode(kernels)

    if kernels == "metamath":
        from . import likelihood_metamath as _src
    else:
        from . import likelihood as _src

    import sys
    pkg = sys.modules[__name__]
    for name in _LIKELIHOOD_CLASSES:
        setattr(pkg, name, getattr(_src, name))

    _KERNELS = kernels


# Make the factory mode and the likelihood-class bindings agree at import time.
# `from .likelihood import *` above binds the top-level PulsarLikelihood /
# GlobalLikelihood / ArrayLikelihood to the matrix classes, and `_kernels._mode`
# starts at "matrix" -- so an explicit config() call is needed to move BOTH to
# the metamath default in lock-step. (Before the flip these agreed only by the
# coincidence of both defaults being "matrix".) Rollback is this one line plus
# the `_KERNELS` default above.
config(kernels=_KERNELS)


__version__ = "0.5"
