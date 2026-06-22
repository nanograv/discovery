Inspecting a model
==================

Once you have assembled a likelihood it is easy to lose track of exactly what
went into it: which signals are present, how many basis functions each one
contributes. Given the way objects are built in discovery, it can also be
difficult to figure out what parts of a ``Likelihood`` object correspond to
which mathematical terms.

For this reason, discovery offers a few ways to visualize the model, and
also offers some directions to the user on where to find pieces of the model
they may want to use.

A first look
------------

.. code-block:: python

   import discovery as ds
   import discovery.recipes as ds_recipes

   psr   = ds.Pulsar.read_feather("B1855+09.feather")
   model = ds_recipes.full_rn(psr)

   print(model.summary())

produces a snapshot like::

   discovery model summary   (backend: matrix, float64)
   ==============================================================================
   PulsarLikelihood - B1855+09   (7758 TOAs)
   ------------------------------------------------------------------------------
   signal                kind                              basis           free
   measurement           white noise                       -                  0
       . B1855+09_430_ASP_efac  (fixed)
       ...
   ecorrGP               GP, fixed prior                   7758 x 360         0
   timingmodel           GP, fixed prior                   7758 x 166         0
   rednoise              GP, variable prior                7758 x 60          2
       + B1855+09_rednoise_log10_A  [-20, -11]
       + B1855+09_rednoise_gamma  [0, 7]
   ==============================================================================
   pulsars: 1   free params: 2   fixed: 8   common: 0
   total basis dimension: 586

Each signal row shows its name, what kind of object it is, the shape of its
design (basis) matrix as *(number of TOAs) x (number of basis functions)*, and
how many free parameters it carries. By default both are listed beneath each
signal: the free parameters with their uniform prior ranges, and the fixed
parameters (e.g. white noise pinned from a noise dictionary) with a ``(fixed)``
marker (see *Toggles* below to change this).

Multiple pulsars
----------------

For a :class:`~discovery.GlobalLikelihood` or
:class:`~discovery.ArrayLikelihood` the summary prints one block per pulsar,
followed by a **common / correlated signals** section for shared red noise,
correlated (e.g. Hellings–Downs) processes, and external deterministic
signals. Correlated signals report their overlap-reduction function and the
number of pulsars they span.

Toggles
-------

The free- and fixed-parameter listings are independent switches, and an
optional ``access`` column prints each signal's live handle (see below)::

    model.summary(show_fixed=False)     # free parameters only (compact)
    model.summary(show_free=False, show_fixed=True)   # just the fixed list
    model.summary(show_access=True)     # add the signals[i] / commongp column

Visualizing the kernel
----------------------

``model.tree()`` draws the model as a tree — how the signals stack into the
covariance — with each signal's *live handle* alongside it::

    PulsarLikelihood  B1855+09  (7758 TOAs)    C = N + Σ FΦFᵀ
    ├─ measurement    white noise                        signals[1]
    ├─ ecorrGP        GP, fixed prior        7758×360    signals[2]
    ├─ timingmodel    GP, fixed prior        7758×166    signals[3]
    └─ rednoise       GP, variable prior     7758×60     signals[4]

Those handles are real: ``model.signals[4]`` *is* the red-noise component, and
``.F`` (its basis) / ``.Phi`` (its prior) / ``.index`` (its coefficient slice)
are returned by reference — no copies. For a multi-pulsar model the per-pulsar
handles are ``psls[k].signals[i]``, and shared signals carry ``commongp`` /
``globalgp`` with each pulsar's slice into the stacked basis.

If you've ever wondered what all of the ``.N.N.N`` structure
corresponds to — ``model.tree(literal=True)`` mirrors
the *actual nested kernel* that the likelihood computes with
(``model.N``, ``model.N.N``, …). With the default
``concat=True`` the constant GPs are fused into a single Woodbury layer, shown
with their column slices::

    ├─ + rednoise       GP, variable prior     7758×60     →  N
    ├─ + [ecorrGP, timingmodel] fused  →  N.N.F
    │  ├─ ecorrGP        N.N.F[:, 0:360]
    │  └─ timingmodel    N.N.F[:, 360:526]
    └─ white noise  →  N.N.N

Each signal object also has an informative ``repr``::

    >>> model.signals[4]
    <VariableGP 'rednoise' | basis 7758×60 | params: B1855+09_rednoise_log10_A, …_gamma>

Other formats
-------------

* ``model.summary(to_stdout=True)`` / ``model.tree(to_stdout=True)`` print
  instead of returning the string.
* ``model.summary_frame()`` returns a :class:`pandas.DataFrame` with one row
  per signal — convenient for programmatic checks or notebook display.
* In a Jupyter notebook, displaying the model renders the composition tree
  (via ``_repr_html_``); ``repr(model)`` gives the same tree in the terminal,
  truncated to the first few pulsars for large arrays.

The summary is pure introspection — it never touches the kernel math — so it
behaves identically under either kernel backend
(``ds.config(kernels='matrix'|'metamath')``).

API
---

.. automodule:: discovery.summary
   :members: summary, summary_frame, summary_html, tree, SignalInfo
   :undoc-members:
