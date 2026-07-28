"""Single-leaf parameter container for fast likelihood evaluation.

A discovery likelihood ``logL`` is called with a parameter dict keyed by name
(~136 entries for a full PTA). Under JAX every dict entry is a separate pytree
leaf, so a jitted ``logL`` has one input binding per parameter and
``value_and_grad`` produces a leaf-per-parameter cotangent. That marshalling --
not the linear algebra -- dominates the GPU forward/grad time.

``Params`` stores every parameter in one flat array (``raw``, the single leaf)
plus a static layout mapping each name to its ``(start, stop, shape)`` in that
array. It is a read-only ``Mapping``, so a dict-based ``logL`` can index it
unchanged, and it registers as a single-leaf JAX pytree, so jit/grad/vmap see
one buffer instead of N.

Backend follows discovery's own switch: arrays are built with ``utils.jnp``
(JAX or NumPy, as configured by ``utils.config``), and the JAX pytree
registration runs only when the JAX backend is active.
"""

import re
import collections.abc

import numpy as np

from . import utils

__all__ = ['Params', 'make_layout']


# discovery encodes an array-valued parameter's shape as a parenthesized suffix
# on its name, e.g. 'B1855+09_red_noise_log10_rho(30)' or 'fourierGP_var(60,60)'.
_SHAPE_RE = re.compile(r'\(([\d,\s]+)\)$')


def _shape_of(name, template=None):
    """Shape of a parameter, read from the parenthesized suffix on its name.

    ``template`` (a {name: value} dict) is an optional fallback for the rare
    array parameters that carry no suffix.
    """
    m = _SHAPE_RE.search(name)
    if m:
        return tuple(int(d) for d in m.group(1).split(','))
    if template is not None:
        return tuple(np.shape(template[name]))
    return ()


def make_layout(names, template=None):
    """Build a hashable layout from an ordered iterable of parameter names.

    Returns ``(layout, size)``: ``layout`` is a tuple of
    ``(name, start, stop, shape)`` entries; ``size`` is the flat length P. The
    layout is hashable so it can serve as JAX pytree aux data (it becomes part
    of the compile cache key, and must not drift between calls).
    """
    entries, pos = [], 0
    for name in names:
        shape = _shape_of(name, template)
        n = 1
        for d in shape:
            n *= d
        entries.append((name, pos, pos + n, shape))
        pos += n
    return tuple(entries), pos


class Params(collections.abc.Mapping):
    """A named, single-leaf parameter container; see the module docstring.

    Construct with :meth:`from_dict` (or :meth:`zeros`); index by name like a
    dict; update functionally with :meth:`update` / :meth:`updates` -- each
    returns a *new* ``Params``, never mutating in place.
    """

    def __init__(self, raw, layout):
        self.raw = raw                          # the single pytree leaf: flat (P,) array
        self.layout = layout                    # static, hashable: ((name, s0, s1, shape), ...)
        self._index = {e[0]: e for e in layout}  # O(1) name -> entry lookup
        self._names = tuple(e[0] for e in layout)

    # --- constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, d, names=None):
        """Build a ``Params`` from a parameter dict.

        ``names`` fixes the column ordering (pass ``logL.params``); it defaults
        to the dict's own key order.
        """
        if names is None:
            names = list(d)
        layout, _ = make_layout(names, template=d)
        raw = utils.jnp.concatenate(
            [utils.jnp.asarray(d[n]).reshape(-1) for (n, _, _, _) in layout])
        return cls(raw, layout)

    @classmethod
    def zeros(cls, names):
        """A ``Params`` of zeros with the layout implied by ``names``."""
        layout, size = make_layout(names)
        return cls(utils.jnp.zeros(size), layout)

    # --- Mapping interface (read-only) -------------------------------------

    def __getitem__(self, name):
        _, s0, s1, shape = self._index[name]     # raises KeyError(name) if absent
        block = self.raw[s0:s1]
        return block.reshape(shape) if shape else block[0]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    # --- functional update -------------------------------------------------

    def update(self, name, value):
        """Return a new ``Params`` with ``name``'s block replaced by ``value``."""
        return self.updates({name: value})

    def updates(self, mapping):
        """Return a new ``Params`` with several blocks replaced, in one pass.

        ``mapping`` is a {name: value} dict (use a dict, not kwargs -- some
        parameter names contain '(' and are not valid identifiers).
        """
        numpy_backend = utils.jnp is np

        if numpy_backend:
            new_raw = np.array(self.raw)              # copy; the original stays intact
        else:
            new_raw = utils.jnp.asarray(self.raw)    # ensure a JAX array to scatter into

        for name, value in mapping.items():
            _, s0, s1, _ = self._index[name]
            v = utils.jnp.asarray(value).reshape(-1)
            if numpy_backend:
                new_raw[s0:s1] = v
            else:
                new_raw = new_raw.at[s0:s1].set(v)

        return Params(new_raw, self.layout)

    # --- conversions / introspection --------------------------------------

    def to_dict(self):
        """A plain dict {name: value} -- the inverse of :meth:`from_dict`."""
        return {name: self[name] for name in self._names}

    @property
    def names(self):
        """Parameter names in flat-array (column) order."""
        return self._names

    @property
    def size(self):
        """Flat-array length P (>= len(self) once any parameter is array-valued)."""
        return self.layout[-1][2] if self.layout else 0

    def __repr__(self):
        return f"Params(size={self.size}, nparams={len(self)})"

    # --- JAX pytree: one leaf (raw), static aux (layout) -------------------

    def tree_flatten(self):
        return (self.raw,), self.layout

    @classmethod
    def tree_unflatten(cls, layout, children):
        return cls(children[0], layout)


# register as a single-leaf pytree when discovery is on the JAX backend
if utils.jnp is not np:
    import jax
    jax.tree_util.register_pytree_node(
        Params, Params.tree_flatten, Params.tree_unflatten)
