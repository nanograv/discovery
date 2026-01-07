from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, deque
from typing import Callable, Dict, List, Union, Iterable, Any, Optional
import functools
import inspect
import ast
from unicodedata import name

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from sympy import Matrix

from . import prior

Array = jax.Array

keepgraph = False

# ===== Graph library =====

# ---- Leaf node types ----

@dataclass
class ArgLeaf:
    name: Optional[str] = None

@dataclass
class ConstLeaf:
    value: Any

@dataclass
class FuncLeaf:
    fn: Callable[..., Any]

@dataclass
class GraphLeaf:
    graph: Graph

Leaf = Union[ArgLeaf, ConstLeaf, FuncLeaf, GraphLeaf]

# ---- Internal node type ----

@dataclass
class Node:
    op: Callable[..., Array]           # JAX-friendly op
    inputs: List[str]                  # names of upstream nodes
    description: Optional[str] = None  # optional description of the node

# Whole graph: name -> Leaf or Node - TODO: it may need to be its own class
Graph = Dict[str, Union[Leaf, Node]]

@dataclass
class Apply:
    pass

def make_leaf(x, name=None) -> Leaf:
    if x is None:
        return ArgLeaf(name=name)
    elif callable(x):
        return FuncLeaf(fn=x)
    elif isinstance(x, OrderedDict):
        return GraphLeaf(graph=x)
    else:
        return ConstLeaf(value=x)


# args takes None and values; the latter will be replaced into ArgLeafs
def fold_constants(graph: Graph, args=[]) -> Graph:
    """
    Collapse any all-constant subgraphs into ConstLeafs.
    Assumes graph is in topological order.
    """
    new_graph: Graph = OrderedDict()
    cache: Dict[str, Union[Array, Callable[..., Any]]] = {}
    args_cache = deque(args)

    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            # replace values for arguments if available
            if len(args_cache) and (argval := args_cache.popleft()) is not None:
                cache[name] = argval
                new_graph[name] = ConstLeaf(value=argval)
            else:
                new_graph[name] = node
        elif isinstance(node, ConstLeaf):
            cache[name] = node.value
            new_graph[name] = node
        elif isinstance(node, FuncLeaf):
            # evaluate function if it doesn't take parameters
            if not getattr(node.fn, 'params', []):
                val = node.fn(params={})
                cache[name] = val
                new_graph[name] = ConstLeaf(val)
            else:
                new_graph[name] = node
        elif isinstance(node, GraphLeaf):
            # constant-fold graph if it doesn't have any argument
            if not any(isinstance(subnode, ArgLeaf) for subnode in node.graph.values()):
                subgraph = fold_constants(node.graph)
                last = subgraph[next(reversed(subgraph))]

                # if the subgraph simplifies to a constant, make a ConstLeaf
                if isinstance(last, ConstLeaf):
                    cache[name] = last.value
                    new_graph[name] = last
                else:
                    new_graph[name] = GraphLeaf(subgraph)
            else:
                new_graph[name] = node
        elif isinstance(node, Node):
            # we're applying a function or graph to a list of inputs; the op will be "apply", but we don't need it
            if node.op is Apply:
                first = new_graph[node.inputs[0]]

                if isinstance(first, GraphLeaf):
                    subargs = [cache[argname] if isinstance(new_graph[argname], ConstLeaf) else None
                            for argname in node.inputs[1:]]
                    subgraph = fold_constants(first.graph, args=subargs)
                    last = subgraph[next(reversed(subgraph))]

                    if isinstance(last, ConstLeaf):
                        cache[name] = last.value
                        new_graph[name] = last
                    else:
                        new_graph[node.inputs[0]] = GraphLeaf(subgraph)
                        new_graph[name] = node
                else:
                    raise NotImplementedError(f"Should we be applying {first} to arguments?")
            # we're applying a function to constant inputs
            elif all(isinstance(new_graph[input], ConstLeaf) for input in node.inputs):
                val = node.op(*[cache[input] for input in node.inputs])
                cache[name] = val
                new_graph[name] = ConstLeaf(val)
            else:
                new_graph[name] = node
        else:
            raise TypeError(f"Unknown node type for {name}: {type(node)}")

    new_graph = prune_graph(new_graph)

    return new_graph


def build_callable_from_graph(graph: Graph):
    output_name = next(reversed(graph.keys()))

    arg_leaves: List[str] = []
    const_values: Dict[str, Array] = {}
    func_leaves: Dict[str, Callable[[Any], Array]] = {}
    graph_leaves: Dict[str, Callable[[Any], Array]] = {}
    nodes: OrderedDict[str, Node] = OrderedDict()

    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            arg_leaves.append(name)
        elif isinstance(node, ConstLeaf):
            const_values[name] = node.value
        elif isinstance(node, FuncLeaf):
            func_leaves[name] = node.fn
        elif isinstance(node, GraphLeaf):
            graph_leaves[name] = build_callable_from_graph(node.graph)
        elif isinstance(node, Node):
            nodes[name] = node
        else:
            raise TypeError(f"Unknown node type for {name}")

    def f(*args, params={}) -> Array:
        env: Dict[str, Array] = {}

        # Get arguments
        for arg, val in zip(arg_leaves, args):
            env[arg] = val

        # Fill constants
        env.update(const_values)

        # Evaluate functions
        for name, fn in func_leaves.items():
            env[name] = fn(params=params)

        for name, fn in graph_leaves.items():
            if not fn.args:
                env[name] = fn(params=params)

        # Evaluate internal nodes in (given) topological order
        for name, node in nodes.items():
            if node.op is Apply:
                first = node.inputs[0]

                if isinstance(graph[first], GraphLeaf):
                    args = [env[input] for input in node.inputs[1:]]
                    env[name] = graph_leaves[first](*args, params=params)
                else:
                    raise NotImplementedError(f"Should we apply {first}?")
            else:
                args = [env[input] for input in node.inputs]
                env[name] = node.op(*args)

        return env[output_name]

    f.args = arg_leaves
    f.params = sorted(set(sum([getattr(fn, 'params', [])
                               for fdict in [func_leaves, graph_leaves]
                               for fn in fdict.values()], [])))

    if keepgraph:
        f.graph = graph
        f.graph.subgraphs = [graph[subgraphname].graph for subgraphname in graph_leaves]

    return f


def prune_graph(graph: Graph,
                output: str = None) -> Graph:
    """
    Given a (possibly folded) graph and possibly an output node name,
    return a new graph containing only the nodes needed to compute
    the output.

    - Keeps all required ConstLeaf / FuncLeaf / Node entries.
    - Preserves original topological order for the kept nodes.
    """
    output_name = next(reversed(graph.keys())) if output is None else output

    required = set([output_name])
    q = deque([output_name])

    while q:
        name = q.popleft()
        node = graph[name]

        # Only Nodes have inputs that we need to traverse
        if isinstance(node, Node):
            for inp in node.inputs:
                if inp not in required:
                    required.add(inp)
                    q.append(inp)

    # Rebuild an OrderedDict with only required nodes,
    # preserving original order
    pruned = OrderedDict(
        (name, node)
        for name, node in graph.items()
        if name in required
    )

    return pruned


def visualize_graph(graph: Graph, fold=False, format='svg', rankdir='TB'):
    """
    Visualize a computational graph using Graphviz.

    Args:
        graph: The computational graph to visualize
        fold: If True, apply constant folding and pruning
        format: Output format ('svg', 'png', 'pdf', etc.)
        rankdir: Graph direction ('TB' top-to-bottom, 'LR' left-to-right)

    Returns:
        graphviz.Digraph object (displays automatically in Jupyter)
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("Please install graphviz: pip install graphviz")

    if fold:
        graph = fold_constants(graph)

    dot = graphviz.Digraph(comment='Computational Graph', format=format)
    dot.attr(rankdir=rankdir)
    dot.attr('node', shape='box', style='rounded,filled', fontname='monospace')

    subgraphs = {}
    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            label = f"{name}: arg"
            dot.node(name, label, fillcolor='lightblue')
        elif isinstance(node, ConstLeaf):
            value_str, _ = _print_array_summary(node.value)
            label = f"{name}: const\\n{value_str}"
            dot.node(name, label, fillcolor='lightgreen')
        elif isinstance(node, FuncLeaf):
            fn_name = getattr(node.fn, '__name__', str(node.fn))
            label = f"{name}: func\\n{fn_name}"
            dot.node(name, label, fillcolor='lightyellow')
        elif isinstance(node, GraphLeaf):
            num_nodes = len(node.graph)
            label = f"{name}: subgraph\\n({num_nodes} nodes)"
            dot.node(name, label, fillcolor='lavender', shape='box3d')
            subgraphs[name] = node.graph
        elif isinstance(node, Node):
            op_name = node.description if node.description else getattr(node.op, '__name__', 'λ')
            label = f"{name}\\n{op_name}"
            dot.node(name, label, fillcolor='white')

    # Add edges
    for name, node in graph.items():
        if isinstance(node, Node):
            for inp in node.inputs:
                dot.edge(inp, name)

    return dot


# helper for print_graph
def _print_array_summary(arr: Any) -> tuple[str, int]:
    if isinstance(arr, (np.ndarray, jnp.ndarray)):
        return (f"array{arr.shape}", arr.size)
    elif isinstance(arr, tuple):
        items = [_print_array_summary(a) for a in arr]
        return (', '.join([item[0] for item in items]), sum([item[1] for item in items]))
    else:
        return (f"{arr}", 0)


def print_graph(graph: Graph, fold=False, name=None) -> None:
    """
    Pretty-print a computational graph, applying constant folding and pruning if simplify=True
    """
    if fold:
        graph = fold_constants(graph)

    print(f"## {'graph' if name is None else name}")

    subgraphs, size = {}, 0
    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            print(f"{name}: arg")
        elif isinstance(node, ConstLeaf):
            const = _print_array_summary(node.value)
            print(f"{name}: const = {const[0]}")
            size = size + const[1]
        elif isinstance(node, FuncLeaf):
            if hasattr(node.fn, 'graph'):
                uname = f"{name}[{np.random.randint(65536):04x}]"
                subgraphs[uname] = node.fn.graph
                print(f"{name}: func = subgraph {uname}")
            else:
                print(f"{name}: func = {node.fn}")
        elif isinstance(node, GraphLeaf):
            uname = f"{name}[{np.random.randint(65536):04x}]"
            subgraphs[uname] = node.graph
            print(f"{name}: graph = subgraph {uname}")
        elif isinstance(node, Node):
            print(f"{name}: node({', '.join(node.inputs)}) = {node.description if node.description else node.op}")
        else:
            print(f"{name}: unknown node type {type(node)}")

    print(f"# total array constants: {size} elements")

    for name, subgraph in subgraphs.items():
        print('', end='\n')
        size = size + print_graph(subgraph, name=f"subgraph {name}")

    return size


def sample_graph(graph: Graph, *args, display=False) -> Graph:
    """
    Make a function from a graph, draw parameters from the prior, and evaluate it.
    """
    f = func(graph, jit=False)

    ret = f(*args, params=prior.sample_uniform(f.params))

    if display:
        print(_print_array_summary(ret)[0])
    else:
        return ret


def func(graph: Graph,
         output: str = None) -> Callable[[Any], Array]:
    """
    Given a computational graph, produce a JAX-jittable function
    that computes the graph output. This first folds constant subgraphs,
    then prunes the graph, then builds the callable.
    """
    if output is not None:
        graph = prune_graph(graph, output)

    return build_callable_from_graph(fold_constants(graph))


# ===== Matrix operations =====


def matrix_inv(amat, params={}):
    if amat.ndim == 1:
        return jnp.diag(1.0 / amat), jnp.sum(jnp.log(jnp.abs(amat)))
    elif amat.ndim == 2:
        if amat.shape[0] == amat.shape[1]:
            return jnp.linalg.inv(amat), jnp.linalg.slogdet(amat)[1]
        else:
            return jnp.eye(amat.shape[1]) * (1.0 / amat)[:, None, :], jnp.sum(jnp.log(jnp.abs(amat)), axis=1)
matrix_inv.args, matrix_inv.params = ['amat'], []


def matrix_solve(amat, b, params={}):
    if amat.ndim == 1:
        if b.ndim == 1:
            return b / amat, jnp.sum(jnp.log(jnp.abs(amat)))
        else:
            return b / amat[:, None], jnp.sum(jnp.log(jnp.abs(amat)))
    else:
        # could do cholesky generically
        return jnp.linalg.solve(amat, b), jnp.linalg.slogdet(amat)[1]
matrix_solve.args, matrix_solve.params = ['amat', 'b'], []


def cholesky_solver(amat, params={}):
    cf = jsp.linalg.cho_factor(amat)

    def solver(b, params={}):
        return jsp.linalg.cho_solve(cf, b)
    solver.args, solver.params = ['b'], []

    if cf[0].ndim == 2:
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cf[0]))))
    else:
        i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(cf[0][:,i1,i2])), axis=1)

    return solver, logdet
cholesky_solver.args, cholesky_solver.params = ['amat'], []


# do it separately if we want the factor

def cholesky_factor(amat, params={}):
    cf = jsp.linalg.cho_factor(amat, lower=True)

    if cf[0].ndim == 2:
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cf[0]))))
    else:
        i1, i2 = jnp.diag_indices(cf[0].shape[1], ndim=2)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(cf[0][:,i1,i2])), axis=1)

    return cf, logdet
cholesky_factor.args, cholesky_factor.params = ['amat'], []

def cholesky_solve(cf, b, params={}):
    return jsp.linalg.cho_solve(cf, b)
cholesky_solve.args, cholesky_solve.params = ['cf', 'b'], []


# ===== Symbolic graph builder =====

@dataclass
class Sym:
    name: str
    builder: "GraphBuilder"

    # Transpose
    @property
    def T(self) -> "Sym":
        return self.builder.node(lambda x: x.T, [self], description=f"{self.name}.T")

    # Matrix multiply
    def __matmul__(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda a, b: a @ b, [self, other], description=f"{self.name} @ {other.name}")

    # Addition
    def __add__(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda a, b: a + b, [self, other], description=f"{self.name} + {other.name}")

    def __sub__(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda a, b: a - b, [self, other], description=f"{self.name} - {other.name}")

    def __radd__(self, other: "Sym") -> "Sym":
        if isinstance(other, (int, float, Array)):
            return self.builder.node(lambda a: other + a, [self], description=f"{other} + {self.name}")
        else:
            return self.__add__(other)

    def __rmul__(self, other: "Sym") -> "Sym":
        if isinstance(other, (int, float, Array)):
            return self.builder.node(lambda a: other * a, [self], description=f"{other} * {self.name}")
        else:
            return self.builder.node(lambda a, b: a * b, [self, other], description=f"{self.name} * {other.name}")

    def solve(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda N, Y: matrix_solve(N, Y), [self, other], description=f"solve({self.name}, {other.name})")

    def inv(self) -> "Sym":
        return self.builder.node(lambda x: matrix_inv(x), [self], description=f"inv({self.name})")

    def dot(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda x, y: x.T @ y, [self, other], description=f"{self.name}.T @ {other.name}")

    def sum(self) -> "Sym":
        return self.builder.node(lambda x: jnp.sum(x), [self], description=f"sum({self.name})")

    # Function or graph application
    def __call__(self, *args: "Sym") -> "Sym":
        return self.builder.node(Apply, [self, *args], description=f"{self.name}({', '.join(arg.name for arg in args)})")

    def pair(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda x, y: (x, y), [self, other], description=f"pair({self.name}, {other.name})")

    # this may lead to problems if we return a list; split may be better
    def split(self) -> tuple["Sym", "Sym"]:
        return (self.builder.node(lambda x: x[0], [self], description=f"{self.name}[0]"),
                self.builder.node(lambda x: x[1], [self], description=f"{self.name}[1]"))

    def __getitem__(self, idx: Union[int, slice]) -> "Sym":
        return self.builder.node(lambda x: x[idx], [self], description=f"{self.name}[{idx}]")

    # deprecate? this may lead to problems if we return a list; split may be better
    def __iter__(self) -> "Iterable[Sym]":
        yield self[0]
        yield self[1]


class GraphBuilder:
    def __init__(self):
        self.graph: Graph = OrderedDict()
        self._counter: int = 0

    def _fresh_name(self, prefix: str = "v") -> str:
        name = f"{prefix}{self._counter}"
        self._counter += 1
        return name

    # --- Leaves ---

    def leaf(self, value_or_fn, name: Optional[str] = None) -> Sym:
        """Create a leaf node for a constant or a fn(runtime)->array."""
        if name is None:
            name = self._fresh_name("leaf")

        if isinstance(value_or_fn, (list, tuple)):
            return [self.leaf(v, f"{name}_{i}") for i, v in enumerate(value_or_fn)]

        self.graph[name] = make_leaf(value_or_fn, name)

        return Sym(name=name, builder=self)

    # --- Generic node creation ---

    def node(self, op: Callable[..., Any], inputs: List[Sym],
             name: Optional[str] = None, description: Optional[str] = None) -> Sym:
        if name is None:
            name = self._fresh_name("n")
        input_names = [s.name for s in inputs]
        self.graph[name] = Node(op=op, inputs=input_names, description=description)
        return Sym(name=name, builder=self)

    def named(self, symbol: Sym, name: str) -> Sym:
        self.graph[name] = self.graph[symbol.name]
        del self.graph[symbol.name]
        symbol.name = name

        return symbol

    # --- Domain-specific ops ---

    def dot(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda X, Y: X.T @ Y, [A, B], name=name, description=f"{A.name}.T @ {B.name}") # was jnp.sum(X * Y)

    def solve(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda N, Y: matrix_solve(N, Y), [A, B], name=name, description=f"solve({N.name}, {Y.name})")

    def cho_factor(self, A: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda amat: cholesky_factor(amat), [A], name=name, description=f'cho_factor({A.name})')

    def cho_solve(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda cf, b: cholesky_solve(cf, b), [A, B], name=name, description=f'cho_solve({A.name}, {B.name})')

    def inv(self, A: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda X: jnp.linalg.inv(X), [A], name=name, description=f'inv({A.name})')


    def array(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *vecs: jnp.array(vecs), args, name=name)

    def hstack(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *vecs: jnp.hstack(vecs), args, name=name)

    def block_diag(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *mats: jsp.linalg.block_diag(*mats), args, name=name)


    def sum_all(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *fts: sum(fts), args, name=name, description=f'sum({",".join(arg.name for arg in args)})')

    def apply(self, A: Sym, *args: Sym, name: Optional[str] = None) -> Sym:
        return self.node(Apply, [A, *args], name=name, description=f'{A.name}({", ".join(arg.name for arg in args)})')

    def pair(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda x, y: (x, y), [A, B], name=name, description=f'pair({A.name}, {B.name})')

    def ntuple(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *xs: tuple(xs), args, name=name, description=f'tuple({", ".join(arg.name for arg in args)})')

    def split(self, AB: Sym, name1: Optional[str] = None, name2: Optional[str] = None) -> tuple[Sym, Sym]:
        return self.node(lambda x: x[0], [AB], name=name1), self.node(lambda x: x[1], [AB], name=name2)

    def stacksolve(self, Nsolve: Sym, y: Sym, F: Sym, name: Optional[str] = None) -> Sym:
        "Combine Nsolve(y) and Nsolve(F) calls, stacking y and F"

        def _stacksolve(solver, y_in, F_in, params={}):
            yF = jnp.hstack([y_in[:, None] if y_in.ndim == 1 else y_in, F_in])

            NmyF, lN = solver(yF, params=params)

            if y_in.ndim == 1:
                Nmy, NmF = NmyF[:, 0], NmyF[:, 1:]
            else:
                Nmy, NmF = NmyF[:, :y_in.shape[1]], NmyF[:, y_in.shape[1]:]

            return ((Nmy, NmF), lN)

        return self.node(_stacksolve, [Nsolve, y, F], name=name, description=f"{Nsolve.name}({y.name}, {F.name})")

    def eval(self, expr: str, args: List[Sym], name: Optional[str] = None) -> Sym:
        raise NotImplementedError("Not ready for prime time...")

        vars = {node.id: None for node in ast.walk(ast.parse(expr, mode='eval')) if isinstance(node, ast.Name)}
        syms = {name: arg.name for name, arg in zip(vars, args)}

        return self.node(eval(f'lambda {", ".join(vars)}: {expr}'), args, description=f'{expr}')


def graph(factory):
    # skip the first arg (the builder)
    argnames = inspect.getfullargspec(factory).args[1:]

    @functools.wraps(factory)
    def makegraph(*args):
        b = GraphBuilder()
        factory(b, *[b.leaf(arg, name) for arg, name in zip(args, argnames)])
        return b.graph

    sig = inspect.signature(factory)
    params = list(sig.parameters.values())[1:]  # Skip first parameter (graph/g/b)
    makegraph.__signature__ = sig.replace(parameters=params)

    return makegraph
