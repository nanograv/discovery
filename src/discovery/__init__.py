"""Discovery"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from .const import *
from .matrix import *
from .prior import *
from .signals import *
from .likelihood import *
from .optimal import *
from .solar import *
from .pulsar import *
from .deterministic import *

__version__ = "0.5"
