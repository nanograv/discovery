"""`metamatrix.graph_params` — pure introspection of a graph's parameter set.

This is what `ArrayLikelihood.clogl_form_resolved` consults to decide whether a
per-pulsar noise solve has free parameters, *before* any likelihood graph is
built. It must never fold or evaluate anything.
"""
from collections import OrderedDict

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
from discovery import metamatrix  # noqa: E402
from discovery import metamath  # noqa: E402


N_TOA = 12


def _const_noise():
    return metamath.NoiseMatrix1D(np.linspace(1.0, 2.0, N_TOA))


def _var_noise(parname="x_efac"):
    base = np.linspace(1.0, 2.0, N_TOA)

    def getphi(params):
        return base * params[parname] ** 2
    getphi.params = [parname]

    return metamath.NoiseMatrix1D(getphi)


def test_constant_noise_solve_reports_no_params():
    assert metamatrix.graph_params(_const_noise().make_solve) == []


def test_variable_noise_solve_reports_its_parameter():
    assert metamatrix.graph_params(_var_noise().make_solve) == ["x_efac"]


def test_woodbury_solve_reports_the_union():
    """A WoodburyKernel over a variable N and a variable P reports both."""
    F = np.random.default_rng(0).normal(size=(N_TOA, 3))

    phi0 = np.array([1.0, 2.0, 3.0])

    def getP(params):
        return phi0 * 10.0 ** params["x_log10_A"]
    getP.params = ["x_log10_A"]

    kernel = metamath.WoodburyKernel(_var_noise(), F, metamath.NoiseMatrix1D(getP))

    assert metamatrix.graph_params(kernel.make_solve) == ["x_efac", "x_log10_A"]


def test_nested_sherman_morrison_reports_ecorr_through_the_graphleaf():
    """The deep-solve regression: an ECORR parameter lives inside a nested
    GraphLeaf of the SM solve. Reporting it is what makes clogl_form='auto'
    correct for a realistic measurement-noise kernel."""
    # 12 TOAs across 4 ecorr epochs (the exposure matrix is 0/1 indicators).
    U = np.zeros((N_TOA, 4))
    for i in range(N_TOA):
        U[i, i % 4] = 1.0

    def getecorr(params):
        return 10.0 ** (2 * params["x_backend_log10_ecorr"]) * np.ones(4)
    getecorr.params = ["x_backend_log10_ecorr"]

    def getwn(params):
        return np.linspace(1.0, 2.0, N_TOA) * params["x_backend_efac"] ** 2
    getwn.params = ["x_backend_efac"]

    kernel = metamath.NoiseMatrixSM(getwn, U, getecorr)

    assert metamatrix.graph_params(kernel.make_solve) == [
        "x_backend_efac", "x_backend_log10_ecorr"]


def test_funcleaf_with_an_attached_graph_reports_that_graphs_params():
    """A FuncLeaf whose callable carries `.graph` contributes that graph's
    parameters too (the `keepgraph` / ffunc-style attachment)."""
    inner = _var_noise("inner_efac").make_solve

    def fn(params=None):
        return 0.0
    fn.params = ["outer_par"]
    fn.graph = inner

    graph = OrderedDict([("leaf", metamatrix.FuncLeaf(fn=fn))])

    assert metamatrix.graph_params(graph) == ["inner_efac", "outer_par"]


def test_cyclic_reference_terminates_via_the_seen_set():
    """A graph that (transitively) refers to itself must not recurse forever."""
    graph = OrderedDict()

    def fn(params=None):
        return 0.0
    fn.params = ["self_par"]
    fn.graph = graph                      # self-reference

    graph["leaf"] = metamatrix.FuncLeaf(fn=fn)
    graph["nested"] = metamatrix.GraphLeaf(graph=graph)

    assert metamatrix.graph_params(graph) == ["self_par"]


def test_repeated_subgraph_is_visited_once_and_reported_once():
    shared = _var_noise("shared_efac").make_solve
    graph = OrderedDict([
        ("a", metamatrix.GraphLeaf(graph=shared)),
        ("b", metamatrix.GraphLeaf(graph=shared)),
    ])

    assert metamatrix.graph_params(graph) == ["shared_efac"]


def test_inspection_does_not_evaluate_or_fold():
    """The leaf callables must never be called: graph_params runs at model
    construction, where no parameter dictionary exists yet."""
    calls = []

    def getphi(params):
        calls.append(params)
        raise AssertionError("graph_params evaluated a leaf")
    getphi.params = ["x_efac"]

    kernel = metamath.NoiseMatrix1D(getphi)

    assert metamatrix.graph_params(kernel.make_solve) == ["x_efac"]
    assert calls == []
