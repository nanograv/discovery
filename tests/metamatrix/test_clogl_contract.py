"""`PulsarLikelihood.clogL` returns a uniform `(params) -> value` callable.

The `make_kernelproduct_gpcomponent` fallback used to return whatever that
method produced -- a graph for a metamath kernel -- while the primary branch
returned an `ffunc`-wrapped callable. `ffunc` is a no-op for callables, so
wrapping the fallback makes the property's contract uniform without changing
the primary branch.
"""
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
from discovery import likelihood_metamath  # noqa: E402
from discovery import metamath  # noqa: E402


N_TOA = 8


class _GraphReturningKernel:
    """Stands in for a kernel whose gpcomponent method returns a GRAPH.

    No production `PulsarLikelihood` assembles one of these today
    (`make_kernelproduct_gpcomponent` lives on `VectorWoodburyKernel`), which is
    exactly why the fallback needs a direct test: it is the branch that would
    hand a raw graph to a caller expecting a callable.
    """
    index = None

    def make_kernelproduct_gpcomponent(self, y):
        # A real metamath graph: the normal log-density of y under a
        # parameter-dependent diagonal noise.
        def getN(params):
            return np.linspace(1.0, 2.0, N_TOA) * params["x_efac"] ** 2
        getN.params = ["x_efac"]

        return metamath.NoiseMatrix1D(getN).make_kernelproduct(y)


class _StubLikelihood(likelihood_metamath.PulsarLikelihood):
    def __init__(self, y):
        self.y, self.delay, self.N = y, [], _GraphReturningKernel()


def test_clogl_fallback_returns_a_callable_with_params():
    y = np.linspace(-1.0, 1.0, N_TOA)
    clogl = _StubLikelihood(y).clogL

    assert callable(clogl)
    assert clogl.params == ["x_efac"]

    value = clogl({"x_efac": 1.3})
    assert np.isscalar(value) or np.asarray(value).shape == ()
    assert np.isfinite(float(value))


def test_clogl_primary_branch_is_unchanged_and_also_uniform(psr):
    """The `make_coefficientproduct` branch was already ffunc-wrapped; the
    contract is the same on both branches."""
    ds.config(kernels="metamath")
    try:
        model = ds.PulsarLikelihood([
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_fourier(psr, ds.powerlaw, components=10, name="rednoise"),
        ])
        clogl = model.clogL
        # Coefficient keys have no entry in the standard prior dictionary;
        # sample the hyperparameters and supply the coefficients directly.
        rng = np.random.default_rng(0)
        coeff_keys = [p for p in clogl.params if "_coefficients(" in p]
        p0 = ds.sample_uniform([p for p in clogl.params if p not in coeff_keys])
        for key in coeff_keys:
            width = int(key[key.index("(") + 1:key.index(")")])
            p0[key] = rng.normal(size=width)
        value = clogl(p0)
    finally:
        ds.config(kernels="matrix")

    assert callable(clogl)
    assert hasattr(clogl, "params")
    assert np.isfinite(float(value))
