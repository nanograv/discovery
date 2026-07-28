"""Three-route model builder shared by the parity tests.

Each parity test compares two metamath routes against the legacy matrix.py
reference:

  - ``matrix``     — stock matrix.py classes via ``likelihood.py``. Reference.
  - ``mh_patched`` — ``likelihood.py`` with ``matrix.*`` monkeypatched to
                     metamath equivalents (``_patch.metamatrix_patch``).
                     Confirms the metamath *kernels* are correct without
                     depending on any rewrite of the likelihood layer.
  - ``mh_native``  — ``likelihood_metamath.py`` exposed via
                     ``ds.config(kernels='metamath')``. Confirms the
                     end-state metamath-native likelihood path.

``build_routes(factory)`` runs ``factory()`` once in each mode and returns
the three resulting models in a dict. Cached method properties on each
model are accessed eagerly inside the route so their internal closures
capture the right module-level state.
"""
import discovery as ds

from ._patch import metamatrix_patch


def _force(model, attrs):
    """Touch named cached_properties so their closures capture the active
    state. Critical for the mh_patched and mh_native routes: properties
    that resolve `matrix.X` lookups inside their body need to be evaluated
    while the patch / config switch is still active. ``getattr`` is wrapped
    in try/except because some properties (e.g.
    ``GlobalLikelihood.conditional`` without a globalgp) legitimately raise
    during evaluation — that's not a test failure, the test just won't use
    that property."""
    for name in attrs:
        try:
            getattr(model, name)
        except Exception:
            pass


def build_routes(factory, force=("logL", "conditional", "clogL",
                                 "sample", "sample_conditional")):
    """Build the model under each of the three routes. Returns
    ``{'matrix': m, 'mh_patched': m, 'mh_native': m}``. ``force`` lists
    cached_properties to access inside each route so their closures bind
    to the correct module-level state. The default touches every property
    a parity test might consume."""
    out = {}
    # Restore whatever the module default is on the way out, rather than
    # assuming it is 'matrix'. After the default flip the module default is
    # 'metamath', so the matrix reference route must set its mode EXPLICITLY --
    # otherwise the "matrix" reference would silently be built under metamath and
    # every parity comparison would become a self-comparison.
    default = ds.config()
    try:
        # matrix reference (explicit, not relying on the module default)
        ds.config(kernels="matrix")
        out["matrix"] = factory()
        _force(out["matrix"], force)

        # metamath via the monkeypatch (legacy likelihood + mh kernels): needs
        # matrix mode active so ds.* binds the likelihood.py classes.
        ds.config(kernels="matrix")
        with metamatrix_patch():
            out["mh_patched"] = factory()
            _force(out["mh_patched"], force)

        # metamath native (likelihood_metamath.py)
        ds.config(kernels="metamath")
        out["mh_native"] = factory()
        _force(out["mh_native"], force)
    finally:
        ds.config(kernels=default)

    return out
