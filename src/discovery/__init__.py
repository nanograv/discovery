"""Discovery"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from discovery.const import *
from discovery.matrix import *
from discovery.prior import *
from discovery.signals import *
from discovery.likelihood import *
from discovery.os import *
from discovery.solar import *
from discovery.pulsar import *
from discovery.deterministic import *

__version__ = "0.2"
