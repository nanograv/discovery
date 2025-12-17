from dataclasses import dataclass
from collections import OrderedDict, deque
from typing import Callable, Dict, List, Union, Iterable, Any, Optional
import inspect

import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from . import prior

Array = jax.Array


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

Leaf = Union[ArgLeaf, ConstLeaf, FuncLeaf]

# ---- Internal node type ----

@dataclass
class Node:
    op: Callable[..., Array]           # JAX-friendly op
    inputs: List[str]                  # names of upstream nodes
    description: Optional[str] = None  # optional description of the node

# Whole graph: name -> Leaf or Node
Graph = Dict[str, Union[Leaf, Node]]


def make_leaf(x, name=None) -> Leaf:
    if x is None:
        return ArgLeaf(name=name)
    elif callable(x):
        return FuncLeaf(fn=x)
    else:
        return ConstLeaf(value=x)


def fold_constants(graph: Graph) -> Graph:
    """
    Collapse any all-constant subgraphs into ConstLeafs.
    Assumes graph is in topological order.
    """
    new_graph: Graph = OrderedDict()
    cache: Dict[str, Union[Array, Callable[..., Any]]] = {}

    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            new_graph[name] = node
        elif isinstance(node, ConstLeaf):
            cache[name] = node.value
            new_graph[name] = node
        elif isinstance(node, FuncLeaf):
            if not getattr(node.fn, 'args', []) and not getattr(node.fn, 'params', []):
                val = node.fn()
                cache[name] = val
                new_graph[name] = ConstLeaf(val)
            else:
                cache[name] = node.fn
                new_graph[name] = node
        elif isinstance(node, Node):
            # check if all inputs are constant in the *new* graph
            all_const = all(
                isinstance(new_graph[inp], ConstLeaf) or (isinstance(new_graph[inp], FuncLeaf) and not getattr(new_graph[inp].fn, 'params', []))
                for inp in node.inputs
            )
            if all_const:
                args = [cache[inp] for inp in node.inputs]
                val = node.op(*args)
                cache[name] = val
                new_graph[name] = ConstLeaf(val)
            else:
                new_graph[name] = node
        else:
            raise TypeError(f"Unknown node type for {name}: {type(node)}")

    return new_graph


def build_callable_from_graph(graph: Graph,
                              outputs: Union[str, Iterable[str], None] = None,
                              jit=True):
    """
    Take a graph (preferably already constant-folded) and produce a function:

        f(params) -> env[output_name]

    where params is an arbitrary pytree consumed by some of the FuncLeaf.fn
    (those that have a non-empty attribute params).
    """

    # figure out the outputs of this graph
    if outputs is None:
        output_names = [next(reversed(graph.keys()))]
    elif isinstance(outputs, str):
        output_names = [outputs]
    else:
        output_names = list(outputs)

    arg_leaves: List[str] = []
    const_values: Dict[str, Array] = {}
    func_leaves: Dict[str, Callable[[Any], Array]] = {}
    nodes: OrderedDict[str, Node] = OrderedDict()

    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            arg_leaves.append(name)
        elif isinstance(node, ConstLeaf):
            const_values[name] = node.value
        elif isinstance(node, FuncLeaf):
            func_leaves[name] = node.fn
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

        for name, fn in func_leaves.items():
            env[name] = fn(params=params) if not getattr(fn, 'args', []) else fn

        # Evaluate internal nodes in (given) topological order
        for name, node in nodes.items():
            args = [env[inp] for inp in node.inputs]

            # if the operation is a function application, we need to decide whether to pass params
            # we will assume that the first argument is the function to be applied
            if callable(args[0]) and getattr(args[0], 'params', []):
                env[name] = node.op(*args, params=params)
            else:
                env[name] = node.op(*args)

        if len(output_names) == 1:
            return env[output_names[0]]
        else:
            return tuple(env[name] for name in output_names)

    # collect params used by functional leaves
    f.args = arg_leaves
    f.params = sorted(set(sum([getattr(fn, 'params', []) for fn in func_leaves.values()],[])))

    # Now wrap with jax.jit
    return jax.jit(f) if jit else f


def prune_graph(graph: Graph,
                outputs: Union[str, Iterable[str], None] = None) -> Graph:
    """
    Given a (possibly folded) graph and one or more output node names,
    return a new graph containing only the nodes needed to compute
    those outputs.

    - Keeps all required ConstLeaf / FuncLeaf / Node entries.
    - Preserves original topological order for the kept nodes.
    """
    if outputs is None:
        output_names = [next(reversed(graph.keys()))]
    elif isinstance(outputs, str):
        output_names = [outputs]
    else:
        output_names = list(outputs)

    required = set(output_names)
    q = deque(output_names)

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


def _print_array_summary(arr: Any) -> str:
    if isinstance(arr, (np.ndarray, jnp.ndarray)):
        return f"array(shape={arr.shape}, dtype={arr.dtype})"
    elif isinstance(arr, tuple):
        return ', '.join([_print_array_summary(a) for a in arr])
    else:
        return f"{arr}"

def print_graph(graph: Graph, simplify=False) -> None:
    """
    Pretty-print a computational graph.
    """
    if simplify:
        graph = prune_graph(fold_constants(graph))

    for name, node in graph.items():
        if isinstance(node, ArgLeaf):
            print(f"{name}: arg")
        elif isinstance(node, ConstLeaf):
            print(f"{name}: const = {_print_array_summary(node.value)}")
        elif isinstance(node, FuncLeaf):
            print(f"{name}: func = {node.fn}")
        elif isinstance(node, Node):
            print(f"{name}: node({', '.join(node.inputs)}) = {node.description if node.description else node.op}")
        else:
            print(f"{name}: unknown node type {type(node)}")


def sample_graph(graph: Graph, *args, display=False) -> Graph:
    f = func(graph, jit=False)

    ret = f(*args, params=prior.sample_uniform(f.params))

    if display:
        print(_print_array_summary(ret))
    else:
        return ret


def func(graph: Graph,
         outputs: Union[str, Iterable[str], None] = None,
         jit=True) -> Callable[[Any], Array]:
    """
    Given a computational graph, produce a JAX-jittable function
    that computes the output of the graph.

    This first folds constant subgraphs, then prunes the graph, then builds the callable.
    """
    folded = fold_constants(graph)
    pruned = prune_graph(folded, outputs=outputs)
    return build_callable_from_graph(pruned, outputs=outputs, jit=jit)

make = func


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

def cholesky_factor(amat, params={}):
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
cholesky_factor.args, cholesky_factor.params = ['amat'], []

def matrix_stack(b, Amat, params={}):
    return jnp.hstack([b[:,None], Amat])
matrix_stack.args, matrix_stack.params = ['b', 'Amat'], []

def matrix_unstack(bAmat, params={}):
    return bAmat[:,0], bAmat[:,1:]
matrix_unstack.args, matrix_unstack.params = ['bAmat'], []


@dataclass
class Sym:
    name: str
    builder: "GraphBuilder"

    # Transpose
    @property
    def T(self) -> "Sym":
        return self.builder.node(lambda x: x.T, [self])

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

    def factor(self) -> "Sym":
        return self.builder.node(lambda x: cholesky_factor(x), [self], description=f"factor({self.name})")

    def inv(self) -> "Sym":
        return self.builder.node(lambda x: matrix_inv(x), [self], description=f"inv({self.name})")

    def dot(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda x, y: x.T @ y, [self, other], description=f"{self.name}.T @ {other.name}")

    def sum(self) -> "Sym":
        return self.builder.node(lambda x: jnp.sum(x), [self], description=f"sum({self.name})")

    # Function application
    def __call__(self, *args: "Sym") -> "Sym":
        return self.builder.node(lambda f, *xs, params={}: f(*xs, params=params), [self, *args], description=f"{self.name}({', '.join(arg.name for arg in args)})")

    def pair(self, other: "Sym") -> "Sym":
        return self.builder.node(lambda x, y: (x, y), [self, other], description=f"pair({self.name}, {other.name})")

    # this may lead to problems if we return a list; split may be better
    def split(self) -> tuple["Sym", "Sym"]:
        return (self.builder.node(lambda x: x[0], [self], description=f"split({self.name})[0]"),
                self.builder.node(lambda x: x[1], [self], description=f"split({self.name})[1]"))

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

    def solve(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda N, Y: matrix_solve(N, Y), [A, B], name=name)

    def factor(self, A: Sym, name: Optional[str] = None) -> Sym:
        return self.node(cholesky_factor, [A], name=name)

    def inv(self, A: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda X: jnp.linalg.inv(X), [A], name=name)

    def stack(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda X, Y: matrix_stack(X, Y), [A, B], name=name)

    def unstack(self, AB: Sym, name1: Optional[str] = None, name2: Optional[str] = None) -> tuple[Sym, Sym]:
        return self.node(lambda X: X[:,0], [AB], name=name1), self.node(lambda X: X[:,1:], [AB], name=name2)

    def dot(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda X, Y: jnp.sum(X * Y), [A, B], name=name)

    def apply(self, A: Sym, *args: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda f, *xs, params={}: f(*xs, params=params), [A, *args], name=name)

    def pair(self, A: Sym, B: Sym, name: Optional[str] = None) -> Sym:
        return self.node(lambda x, y: (x, y), [A, B], name=name)

    def ntuple(self, args: List[Sym], name: Optional[str] = None) -> Sym:
        return self.node(lambda *xs: tuple(xs), args, name=name)

    def split(self, AB: Sym, name1: Optional[str] = None, name2: Optional[str] = None) -> tuple[Sym, Sym]:
        return self.node(lambda x: x[0], [AB], name=name1), self.node(lambda x: x[1], [AB], name=name2)


def graph(factory):
    # skip the first arg (the builder)
    argnames = inspect.getfullargspec(factory).args[1:]

    def makegraph(*args):
        b = GraphBuilder()
        factory(b, *[b.leaf(arg, name) for arg, name in zip(args, argnames)])
        return b.graph

    return makegraph

