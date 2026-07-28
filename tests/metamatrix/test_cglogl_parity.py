"""matrix-vs-metamath parity for `ArrayLikelihood.cglogL`.

`cglogL` -- the conjugate-gradient log-likelihood for very large arrays (CG
solve + stochastic-Lanczos log-det estimator) -- was the one likelihood frontend
with no parity coverage. It is iterative and its log-det is a stochastic
estimate, so machine precision is not expected: `rtol=1e-6` is the honest gate.

WHY THIS SKIPS TODAY. `cglogL` is currently broken on BOTH routes, and the
breakage predates this cleanup (verified against the pre-cleanup commit):

  * its CG/log-det helpers are optional extras (`jaxopt` + `matfree`), undeclared
    in pyproject and absent from the devcontainer -- so `utils` defines neither,
    and `cglogL` raises AttributeError before any math runs;
  * with `matfree` present, the metamath route has no
    `VectorWoodburyKernel.make_kernelterms` (globalgp branch) and returns an
    unwrapped graph (commongp-only branch);
  * the matrix route's globalgp branch dies inside the CG stack on a JAX API
    change (`matrix_transpose` rejects 1-D input).

Repairing that is out of scope for the graph-consistency cleanup (see
docs/metamatrix.md, "cglogL is not currently runnable"). This test is committed
so it starts enforcing parity the moment the path is repaired, rather than
being written from scratch then.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import discovery.recipes as R  # noqa: E402
from discovery import utils  # noqa: E402


# The CG solve and the Lanczos-Hutchinson log-det estimator are optional
# extras (`jaxopt` + `matfree`), imported behind a try/except in `utils`. With
# either missing, `cglogL` is unavailable on BOTH routes and there is nothing
# to compare.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (hasattr(utils, "cgsolve") and hasattr(utils, "make_logdet_estimator")),
        reason="cglogL needs the optional jaxopt + matfree extras"),
]


def _cglogl_values(psrs, p0s):
    out = {}
    for kernels in ("matrix", "metamath"):
        ds.config(kernels=kernels)
        try:
            model = R.intrinsic_rn_plus_global_hd(psrs)
            cglogl = model.cglogL(cgmaxiter=100)
            out[kernels] = [float(cglogl(p0)) for p0 in p0s]
        finally:
            ds.config(kernels="matrix")
    return out


def test_cglogl_matrix_vs_metamath(psrs):
    """The 3-pulsar fixture, `intrinsic_rn_plus_global_hd`, 3 draws."""
    ds.config(kernels="matrix")
    try:
        params = R.intrinsic_rn_plus_global_hd(psrs).cglogL(cgmaxiter=100).params
    finally:
        ds.config(kernels="matrix")

    np.random.seed(20260716)
    p0s = [ds.sample_uniform(params) for _ in range(3)]

    values = _cglogl_values(psrs, p0s)

    assert np.all(np.isfinite(values["matrix"]))
    np.testing.assert_allclose(values["metamath"], values["matrix"], rtol=1e-6)
