# Metamatrix architecture

Design rationale and house rules for the graph kernel path. This is a
**durable design note**, not a living project board.

For day-to-day contributor guidance see the Sphinx
[Metamatrix developer guide](../../metamatrix_dev.md). For the remaining
deletion work see [deletion_checklist.md](deletion_checklist.md).

---

## End state

**`matrix.py` does not exist.** Every kernel operation, GP, and likelihood
component is built on the graph machinery in `metamatrix.py` (the DSL) and
`metamath.py` (the kernel/GP classes that use it). `signals.py` constructs
metamath objects directly. `likelihood.py` composes metamath graphs without
any matrix.py imports.

The current branch is a transitional state. `matrix.py` is still present,
still imported by `signals.py` and `likelihood.py`, and still functionally
correct — which is the only reason it is useful: **it is the oracle against
which the metamath replacements are tested**. The parity suite under
`tests/metamatrix/` exists to certify, row by row, that every method the
likelihood layer calls produces the same numerical result via the metamath
graph as via the matrix.py closure. When the parity suite covers every
path that any production user exercises, matrix.py gets deleted.

The motivation is not "make metamath also work" — it is "stop maintaining
matrix.py." matrix.py is costly to maintain (see the variant table below),
has known gaps (no `make_kernelproduct_gpcomponent` for non-VectorWoodbury,
no CG-MDL logdet outside one class, no `make_conditional` on
`VectorWoodburyKernel_varP`, etc.), and adding a feature means touching 4-6
sibling classes. The graph-based rewrite eliminates that combinatorial
maintenance surface entirely.

## Why this refactor exists

`matrix.py` evolved into a combinatorial explosion. For the Woodbury kernel
`Σ = N + F P F^T`, each of `N`, `F`, and `P` can be either fixed-at-trace-time
("constant") or parameter-dependent ("variable"). The current file enumerates
the variants by hand:

| Class | N | F | P |
|---|---|---|---|
| `WoodburyKernel_novar` | const | const | const |
| `WoodburyKernel_varP`  | const | const | var   |
| `WoodburyKernel_varN`  | var   | const | const |
| `WoodburyKernel_varNP` | var   | const | var   |
| `WoodburyKernel_varFP` | const | var   | var   |
| `VectorWoodburyKernel_varP` | const | const | var (per-pulsar) |
| ... | | | |

Plus per-class duplication of:
`make_kernelproduct`, `make_kernelproduct_vary`,
`make_kernelterms`, `make_kernelterms_vary`,
`make_kernelsolve`, `make_kernelsolve_vary`,
`make_kernelsolve_simple`, `make_solve_1d`, `make_solve_2d`,
`make_kernelproduct_gpcomponent`, `make_sample`...

Each of these does the same math, branching on whether things are callables vs
arrays, threading `params` only where needed, and pre-baking everything else.
The result is ~2000 lines of `matrix.py` where any nontrivial change has to be
mirrored across 4–6 sibling classes. New features (decentering, additives,
ExtSignals, CG-MDL logdet) get added to *one* class and the rest go stale.

**The metamatrix refactor exists to collapse this into a single generic path.**

## The core idea

Express every kernel operation as a **computation graph**, not a Python
closure. The graph is built once, declaratively, in terms of symbolic
operands. At runtime:

- **`fold_constants`** walks the graph and evaluates every node whose inputs
  are all constants. A "variable" thing becomes a constant simply by being a
  leaf with no free params — same code path.
- **`prune_graph`** removes nodes the requested output doesn't depend on.

This means the same source for `woodbury(y, Nsolve, F, Pinv)`:

- if `Nsolve`, `F`, `Pinv` are all constant → folds entirely into a single
  ConstLeaf at trace time. Equivalent to `WoodburyKernel_novar`.
- if `Pinv` is parameter-dependent → fold stops at the `cho_factor` node;
  everything upstream that doesn't depend on `Pinv` is still pre-baked.
  Equivalent to `WoodburyKernel_varP`.
- if `Nsolve` *and* `Pinv` are param-dependent → almost nothing folds;
  graph is evaluated end-to-end at runtime. Equivalent to `WoodburyKernel_varNP`.
- if `F` is callable → fold stops at the `F`-using nodes; same logic falls out.

**There is no "fixed N case" vs "variable N case" in the source.** There is
one expression. Folding decides what runs at trace time vs runtime.

This is why metamatrix's `mh.WoodburyKernel` doesn't have suffixes. There is
no `_var*` — there is just `WoodburyKernel`. The graph adapts.

## Graph primitives

`metamatrix.py` defines:

- **Leaves**
  - `ArgLeaf(name)` — runtime argument (e.g. residual vector passed each call).
  - `ConstLeaf(value)` — fixed array, baked into the graph.
  - `FuncLeaf(fn)` — callable that takes `params` and returns an array.
    Carries `fn.params` so the resulting graph callable knows what params to
    accept.
  - `GraphLeaf(graph)` — a nested graph, called via `Apply`.

- **Nodes**
  - `Node(op, inputs, description)` — a JAX-friendly op applied to upstream
    nodes/leaves by name.

- **The DSL** (`GraphBuilder`, `Sym`) — Python-level shorthand. `Sym`
  overloads `@`, `*`, `+`, `-`, `__call__`, `.T`, `.solve`, `.inv`, `.dot`,
  `.split`, `__iter__` so the math reads naturally:
  ```python
  @mm.graph
  def woodbury(g, y, Nsolve, F, Pinv):
      Nmy, lN = Nsolve(y)            # Apply on a GraphLeaf
      NmF, _  = Nsolve(F)
      FtNmy   = g.dot(NmF, y)        # F^T N^-1 y
      FtNmF   = g.dot(F, NmF)
      Pm, lP  = Pinv                 # destructure a 2-tuple result
      cf, lS  = g.cho_factor(Pm + FtNmF)
      ...
      logp = -0.5 * (g.dot(y, Nmy) - g.dot(FtNmy, mu)) - 0.5 * ld
  ```
  No `if callable(F): ...` branches. No `params = ... if var else None`.
  Just the math.

- **`mm.func(graph)`** → builds a JAX-jittable callable `f(*args, params={})`.
  Walks the (folded, pruned) graph and emits ops. Carries `f.params` (gathered
  from FuncLeaf / GraphLeaf params).

## Method contracts under metamatrix

Every kernel method that used to return a closure now returns a **graph**.
Where matrix.py exposed `make_kernelproduct(y) -> callable`, metamath exposes
`make_kernelproduct(y) -> Graph`. The caller composes graphs and converts to
a callable only at the outermost boundary (`ffunc` / `mm.func`) — i.e.,
inside `likelihood.py`'s `logL`, `conditional`, `clogL` cached_properties.

Concretely:

| Method | Returns | Notes |
|---|---|---|
| `NoiseMatrix.make_solve` | graph | input `y` → `(Nmy, lN)` |
| `NoiseMatrix.make_inv`   | graph | () → `(Nm, lN)` |
| `WoodburyKernel.make_solve` | graph (pruned to 'solve') | |
| `WoodburyKernel.make_kernelproduct(y)` | graph | scalar logp |
| `WoodburyKernel.make_kernelsolve(y, T)` | callable wrapping a graph | matrix.py contract for callsites that expect `ksolve(params)` |
| `WoodburyKernel.make_conditional(y)` | graph (pruned to 'cond') | |
| `VectorWoodburyKernel.make_kernelproduct(ys)` | graph | over a list of pulsars |
| `VectorWoodburyKernel.make_kernelproduct_gpcomponent(...)` | **graph** (target) | currently missing — see below |
| `make_sample` | plain callable | PRNG threading doesn't fit the graph DSL cleanly; exception |

Subgraphs compose naturally. When `WoodburyKernel.make_kernelproduct` needs
the inner noise solve, it doesn't *call* `self.N.make_solve()` to get an
array — it embeds `self.N.make_solve` as a `GraphLeaf`. The whole tree is
one composite graph that `fold_constants` simplifies in one pass.

## What this means for porting from matrix.py

The wrong way to port a matrix.py method to metamath:

```python
# DON'T — transliterates the variable/constant branching
def make_kernelproduct_gpcomponent(self, ys, ...):
    NmFs = [N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)]   # const-path only
    FtNmFs = [F.T @ NmF for F, NmF in zip(self.Fs, NmFs)]
    FtNmF = jnparray(FtNmFs)
    ...
    def kernelproduct(params): ...
    return kernelproduct
```

This reproduces the matrix.py constraint ("N and F must be constant") and
adds nothing — it's matrix.py written in a different file. Every variant
class still implicitly exists; it's just been folded into runtime errors and
"don't pass variable N" assumptions.

The right way:

```python
# DO — express the math symbolically; let folding handle const-vs-var
@mm.graph
def vectorgpcomponent(g, ys, Nsolves, Fs, Pinv, reparams=(), additives=(), extsignals=()):
    # per-pulsar trace-time-or-runtime solves (graph picks)
    NmFs   = [Nsolve(F) for Nsolve, F in zip(Nsolves, Fs)]
    Nmys   = [Nsolve(y) for Nsolve, y in zip(Nsolves, ys)]
    FtNmFs = [g.dot(F, NmF[0]) for F, NmF in zip(Fs, NmFs)]
    NmFtys = [g.dot(NmF[0], y) for NmF, y in zip(NmFs, ys)]
    ytNmys = [g.dot(y, Nmy[0]) for y, Nmy in zip(ys, Nmys)]
    ldNs   = [Nmy[1] for Nmy in Nmys]
    ...
```

When `Nsolves[i]` is a constant (noisedict supplied), the `Nsolve(F)` subgraph
folds entirely. When it's parameter-dependent (free efacs), the same source
runs at runtime. **One code path, both cases.** No `solve_2d` vs
`make_solve_2d` split. No `_var` class. No `if callable(F)`.

## House rules for `metamath.py`

**We build graphs. We do not build closure factories.**

Every kernel-math method (`make_kernelproduct`, `make_solve`,
`make_conditional`, `make_kernelproduct_gpcomponent`, ...) returns a
metamatrix graph (a dict produced by an `@mm.graph` function), not a Python
callable. Composition with sub-objects (`N.make_solve`, `P.make_inv`, an
ExtSignal's coeff map, a reparam, `self.means`, ...) happens by passing them
into the graph as leaves — `GraphLeaf` for nested graphs, `FuncLeaf` for
param-dependent callables, `ConstLeaf` for arrays. `fold_constants` then
decides what runs at trace time vs runtime. That is the entire point.

Hard rules for new code in `metamath.py`:

1. **Do not call `mm.func(...)` inside a kernel method to evaluate a subgraph
   at construction time and capture the result in a closure.** That defeats
   folding/pruning across the composite graph and reinvents the matrix.py
   const-vs-var split this module exists to eliminate.
2. **Do not write `mm.func(subgraph)(args, params={})` to "materialize" a piece
   at trace time.** If a subexpression is constant, folding will bake it. If
   it isn't, it must stay in the graph.
3. **Do not branch on `callable(x)` / `isinstance(x, dict)` /
   `hasattr(self, 'prior')`** to pick between a constant path and a variable
   path. Express the math once; let folding pick.
4. **Do not introduce helpers like `_materialize` for kernel-math code.**
   `_materialize` exists only for `make_sample`, the documented exception
   (PRNG threading doesn't compose cleanly through the DSL). No other method
   gets that escape hatch.
5. **`mm.func` is called exactly once, at the outermost boundary** — in
   `likelihood.py`'s `logL` / `conditional` / `clogL` cached_properties. Not
   inside `metamath.py` kernel methods.

If you're translating something from `matrix.py` and find yourself reaching
for `mm.func`, stop. You are writing `matrix.py` in a different file. See
"What this means for porting from matrix.py" above for the right pattern.

## Practical guidance

1. **Stop thinking "constant" vs "variable" when writing metamath methods.**
   They are not two cases; they are two folding outcomes of one expression.
2. **Don't gate features on "N and F are constant."** If `make_kernelproduct_gpcomponent`
   needs to multiply `F^T N^-1 F`, it writes `g.dot(F, Nsolve(F))` and lets
   folding handle the rest. The runtime cost when N is variable is exactly the
   cost of actually solving it — there is no extra overhead because of the
   graph layer (folded paths bake out completely).
3. **Reparams / additives / extsignals are just leaves.** A reparam is a
   FuncLeaf returning `(c, ldL)`. An additive is a FuncLeaf returning a
   coefficient correction. An ExtSignal contributes a precomputable
   trace-time block (its `Fs` are constant) and a coefficient-map FuncLeaf.
   These compose into the same graph without bespoke handling.
4. **When porting matrix.py code, identify the math, then re-express it
   symbolically through `GraphBuilder`.** Discard the bookkeeping that exists
   only to manage the constant/variable split.
5. **Sampling (`make_sample`) is the documented exception** — `jax.random`'s
   key-threading model doesn't compose cleanly through the graph DSL, so
   `make_sample` returns a plain callable. This is the only kernel method
   that does.

## Composition example: decentering

To see how this scales, consider what `ArrayLikelihood.clogL` with
`decenter=True` should look like in metamath:

```python
@mm.graph
def gpcomponent_with_reparams(g, ys, Nsolves, Fs, Pinv, decenter, ...):
    # ... per-pulsar solves as above ...

    # decentering is a reparam on c. In the graph: it's a node returning (c, ldL).
    c0 = g.fold_params_to_array(...)   # xi -> c via index map
    c, ldL = decenter(c0, FtNmFs, NmFtys, Pinv)   # ldL is just another scalar node

    logpr = ...   # prior on c (post-reparam)
    c_total = c + sum(add(params) for add in additives)
    quad = -0.5 * jnp.einsum('ij,ijk,ik', c_total, FtNmF, c_total) + ...
    logp = quad + logpr + ldL + extcontrib(c_total, extsignals)
```

The decenter transform itself is a small graph that consumes precomputed
`FtNmFs`, `NmFtys`, and `Pinv` — and again, whether `Pinv` is "fixed"
(because all GP params are pinned) or variable doesn't change the source.
The graph adapts.

This is the eventual payoff: matrix.py's `WoodburyKernel_varP.make_kernelproduct_gpcomponent`
becomes one symbolic expression in metamath, valid for every combination of
fixed/variable N/F/P. **The 18 paths collapse to one.**

## Migration plan (historical)

Phases 0–4 (shared substrate, factory, likelihood_metamath, parity coverage,
carry-overs) are **done**. The remaining work is deletion of the legacy
`matrix.py` / `likelihood.py` oracle path.

See:

- [Deletion checklist](deletion_checklist.md) — ordered steps for the cutover
- [Parity coverage](parity_coverage.md) — constructor → test map
- Sphinx [Metamatrix developer guide](../../metamatrix_dev.md) — current ops docs

Do not use older “current state / next session” notes in git history as
authoritative; the checklist and Sphinx guides supersede them.
