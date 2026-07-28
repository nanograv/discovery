# Metamatrix: developer guide

This page is for **contributors** extending or maintaining the graph kernel
path. For day-to-day model building see [Metamatrix user guide](metamatrix).

---

## 1. Goals and current state

**Long-term goal.** One kernel implementation: the graph path in
`metamatrix.py` (DSL) + `metamath.py` (kernel/GP classes). The legacy
`matrix.py` / `likelihood.py` stack is a numerical **oracle** during
transition, not a second product line.

**Current state.**

- Default: `discovery.config()` → `"metamath"`.
- Production code builds kernels through `_kernels` factory; signals/deterministic
  do not hard-code matrix classes.
- `likelihood_metamath.py` composes metamath kernels directly (no `matrix`
  import for the hot path).
- `tests/metamatrix/` certifies matrix ↔ metamath numerical agreement on the
  topologies production recipes and example notebooks use.
- **Not yet done:** delete `matrix.py` / legacy `likelihood.py`, collapse the
  factory to metamath-only, replace route-vs-route parity with goldens. That is
  a deliberate follow-up after external review of the dual-path PR.

Motivation in one sentence: `matrix.py` enumerates const/var combinations of
$N$, $F$, $P$ as sibling classes (~4–6 copies of each method); the graph path
writes the math once and lets **constant folding** specialise.

---

## 2. Module map

| Module | Role |
|---|---|
| `src/discovery/metamatrix.py` | Graph DSL: leaves, nodes, `@graph`, `fold_constants`, `prune_graph`, `func()`. |
| `src/discovery/metamath.py` | Kernel/GP classes built on the DSL. Methods return **graphs** (not closures). |
| `src/discovery/utils.py` | Path-neutral substrate: numerical `config(backend=…)`, `jnp`/`jsp`/factor aliases, markers (`Kernel`, `GP`, `ConstantGP`, `VariableGP`, `GlobalVariableGP`, `ExtSignal`), Sherman–Morrison index helpers, `make_uind`. |
| `src/discovery/_kernels.py` | Constructor factory. `set_mode("matrix"\|"metamath")` selects classes. Canonical metamath map: `_METAMATH`. |
| `src/discovery/_kernel_switch.py` | **Tests only:** temporarily patch `matrix.*` names to metamath classes so the legacy likelihood can run on new kernels. Shares the map with `_kernels` so they cannot drift. |
| `src/discovery/measurement_noise.py` | Collapsed `makenoise_measurement{,_simple}` (array vs callable chooses fixed vs variable without `_novar`/`_var` classes). Re-exported from `signals`. |
| `src/discovery/signals.py` / `deterministic.py` | Model builders; kernels via factory + utils markers. |
| `src/discovery/likelihood.py` | Legacy likelihoods on `matrix.py` (oracle). |
| `src/discovery/likelihood_metamath.py` | Graph-native `PulsarLikelihood` / `GlobalLikelihood` / `ArrayLikelihood`. |
| `src/discovery/transport.py` | Boundary module: free-standing decentering / reparam objects. May call `metamatrix.func` at **construction** to bake constants. |
| `src/discovery/summary.py` | Model tables including coefficient-frontend treatment. |
| `src/discovery/recipes/` | Importable model zoo shared by tests and the cookbook. |
| `tests/metamatrix/` | Parity and contract suite. |
| `tests/single_precision/` | float32 / reference+delta / projection tests. |

Top-level switch (`src/discovery/__init__.py`):

1. Sets `_kernels` mode.
2. Rebinds `discovery.PulsarLikelihood` / `GlobalLikelihood` / `ArrayLikelihood`
   to the chosen module.
3. At import, calls `config(kernels="metamath")` so factory mode and class
   bindings stay in lock-step (the star-import from `likelihood` would otherwise
   leave matrix classes bound).

---

## 3. Graph DSL (`metamatrix.py`)

### 3.1 Leaves and nodes

- **`ArgLeaf(name)`** — runtime argument (e.g. residual each call).
- **`ConstLeaf(value)`** — fixed array, baked into the graph.
- **`FuncLeaf(fn)`** — `params → array`, carries `fn.params`.
- **`GraphLeaf(graph)`** — nested graph applied as a unit.
- **`Node(op, inputs, description)`** — JAX-friendly op on upstream names.

### 3.2 Building graphs

```python
import discovery.metamatrix as mm

@mm.graph
def woodbury(g, y, Nsolve, F, Pinv):
    Nmy, lN = Nsolve(y)
    NmF, _ = Nsolve(F)
    FtNmy = g.dot(NmF, y)       # F^T N^{-1} y
    FtNmF = g.dot(F, NmF)
    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    # ...
    # Last created node is the default graph output.
```

`Sym` overloads (`@`, `*`, `+`, `.T`, `.dot`, …) so math reads naturally.
**No** `if callable(F)` branches for const vs var.

### 3.3 Materialisation

```python
f = mm.func(graph)           # jittable f(*args, params={})
# f.params collected from FuncLeaf / GraphLeaf
```

`fold_constants` evaluates every node whose inputs are all constant.
`prune_graph` drops nodes unused by the requested output.

**Rule:** call `mm.func` **once**, at the outer boundary (`likelihood_*`
cached properties via `ffunc`, or `transport.py` bake). Kernel methods in
`metamath.py` return graphs and embed subgraphs as leaves.

### 3.4 Precision helpers

Reuse existing patterns from `woodbury` / fused HD paths:

- `pin_f64`, `combine_f64`, `combine_logp_f64` for float32 work with float64
  scalar assembly.
- Do not invent new precision conventions without tests under
  `tests/single_precision/`.

### 3.5 Visualisation

`mm.visualize_graph(graph)` aids debugging of fold/prune behaviour.

---

## 4. House rules for `metamath.py`

We build **graphs**, not closure factories.

1. **Do not** call `mm.func(...)` inside a kernel method to evaluate a subgraph
   at construction and capture it in a closure — that defeats cross-graph folding.
2. **Do not** write `mm.func(subgraph)(args, params={})` to “materialize early.”
   If it is constant, folding bakes it; if not, it stays live.
3. **Do not** branch on `callable(x)` / `isinstance` / `hasattr` to pick constant
   vs variable paths. Express the math once; folding specialises.
4. **`make_sample` is the documented exception** — PRNG key threading does not
   compose cleanly through the DSL, so it returns a plain callable.
5. Porting from `matrix.py`: identify the math, re-express it with
   `GraphBuilder`; discard bookkeeping that only existed to manage const/var.

### Right vs wrong port

```python
# WRONG — freezes the matrix.py assumption "N and F must be constant"
NmFs = [N.solve_2d(F) for N, F in zip(self.Ns, self.Fs)]
def kernelproduct(params): ...
return kernelproduct

# RIGHT — one symbolic path
@mm.graph
def vectorgpcomponent(g, ys, Nsolves, Fs, Pinv, ...):
    NmFs = [Nsolve(F) for Nsolve, F in zip(Nsolves, Fs)]
    ...
```

---

## 5. Method contracts

Where matrix returned callables, metamath returns graphs (unless noted).

| Method | Returns | Notes |
|---|---|---|
| `NoiseMatrix.make_solve` | graph | $y \mapsto (N^{-1}y, \log\det N)$ |
| `NoiseMatrix.make_inv` | graph | $\mapsto (N^{-1}, \log\det)$ |
| `WoodburyKernel.make_kernelproduct(y)` | graph | scalar log density |
| `WoodburyKernel.make_kernelsolve(y, T)` | callable wrapping graph | matrix-era call sites |
| `WoodburyKernel.make_conditional(y)` | graph | |
| `VectorWoodburyKernel.make_kernelproduct(ys)` | graph | multi-pulsar |
| `VectorWoodburyKernel.make_kernelproduct_gpcomponent` | graph | **cross-form** `clogL` |
| `VectorWoodburyKernel.make_residualproduct` | graph | **residual-form** `clogL` |
| `make_sample` | plain callable | PRNG exception |

Sub-objects (`N.make_solve`, `P.make_inv`, ExtSignal coeff maps, reparams, means)
enter as leaves so one `fold_constants` pass sees the full tree.

---

## 6. Old → new class mapping

Graph folding dissolves const/var and most 1D/2D/vector splits:

| `matrix.py` | metamath / signals | Note |
|---|---|---|
| `WoodburyKernel_{novar,varP,varN,varNP,varFP}` | `metamath.WoodburyKernel` | folding |
| `VectorWoodburyKernel_varP` | `VectorWoodburyKernel` | includes `make_conditional` |
| (ad hoc) | `GlobalWoodburyKernel` | first-class HD kernel |
| `NoiseMatrix1D_{novar,var}` | `NoiseMatrix1D` | array or callable |
| `NoiseMatrix2D_{novar,var}` | `NoiseMatrix2D` | |
| `NoiseMatrixSM_{novar,var}` | `NoiseMatrixSM` | indexed SM |
| `VectorNoiseMatrix{1D,2D,12D}_var` | `NoiseMatrix1D/2D/12D` | vector distinction dissolved |
| `CompoundGP`, `VectorCompoundGP` | `CompoundGP` | mixed-Φ marginal path supported |
| `matrix.CompoundGlobalGP` | `signals.CompoundGlobalGP` | backend-agnostic |
| `CompoundDelay` | `CompoundDelay` | same contract |

Canonical name→class map: `_kernels._METAMATH`. Matrix mode falls through to
`matrix.*` at call time so the test monkeypatch still works.

---

## 7. Likelihood assembly (`likelihood_metamath.py`)

### 7.1 Two assemblies

Cached helpers avoid call-order bugs (whichever property ran first used to
mutate `self.vsm` / `self.ys` differently):

| Helper | Used by | Behaviour |
|---|---|---|
| `_marginal_assembly` | `logL`, `cglogL`, `conditional` | commongp only; attaches `P_ref` when `reference=` set |
| `_coefficient_assembly` | `clogL` | folds globalgp in; **does not** consult `reference=` |

Assigning `reference` after construction invalidates dependent caches.

### 7.2 Shadow guard

`PulsarLikelihood(concat=False)` with ≥2 variable GPs raises unless
`marginalize_all_but_last=True`. Chained construction overwrites `.index` each
iteration; without the guard, every variable GP but the last was silently
marginalized. Same guard is on the legacy likelihood for consistency.

### 7.3 `clogL` forms

Shared plumbing lives on `VectorWoodburyKernel._coefficient_leaves`
(prior / fold / means). Two products:

- **Cross** — `vectorgpcomponent` / `make_kernelproduct_gpcomponent`: builds
  $F^\top N^{-1} F$ and cross terms with ExtSignals **separately**.
- **Residual** — `vectorresidualcomponent` / `make_residualproduct`: single
  residual $r = y - Fc - \sum F_{\mathrm{ext}}c_{\mathrm{ext}}$, one $N$-solve.

Routing: `clogl_form="auto"|"cross"|"residual"`. Auto uses pure introspection
via `metamatrix.graph_params` (union of `.params` over a graph; **must not**
fold or evaluate). Residual branch is metamath-only (`_kernels.require_metamath`).

**Bug (open):** cross form drops inter-ExtSignal cross-terms for ≥2
non-orthogonal ExtSignals. Residual is correct. See
[issue #137](https://github.com/nanograv/discovery/issues/137).

### 7.4 Transport composition

`ArrayLikelihood(transport=...)` or `decenter=True` (sugar) composes a reparam
**before** user transforms and before the resolved `clogL` product. Shape and
key order are checked eagerly against the coefficient assembly.
`decenter` and `transport` are mutually exclusive.

Legacy `likelihood.py` still uses an in-likelihood decenter closure; only the
metamath path uses `transport.py`.

### 7.5 `cglogL`

Broken on **both** routes today (optional `jaxopt`+`matfree` undeclared;
metamath missing `VectorWoodburyKernel.make_kernelterms` for globalgp;
matrix globalgp hits JAX API drift). Parity test
`tests/metamatrix/test_cglogl_parity.py` is committed and **skips**. Repair is
separate work.

---

## 8. Transport (`transport.py`)

### 8.1 Role

Boundary module: may bake with `metamatrix.func` at construction; participates
in likelihood graphs only through the FuncLeaf / reparam contract
(`xi → (q, ldJ)`). Kernel methods stay graph-pure.

### 8.2 Math (joint / frozen-noise transport)

Per pulsar, blocks $b=1..B$ with bases $W_b$ and conditioner precisions
$p_b(\theta)$:

$$
\begin{aligned}
W &= [W_1|\cdots|W_B], \\
G_0 &= W^\top N_0^{-1} W, \quad
b_0 = W^\top N_0^{-1} r_0, \\
A(\theta) &= G_0 + \mathrm{diag}(p(\theta)), \\
q &= \mu + L^{-\top}\xi, \quad A = LL^\top, \\
\mu &= A^{-1}b_0\ \text{(if center)}, \quad
\mathrm{ldJ} = -\sum_i \log L_{ii}.
\end{aligned}
$$

$\mu$ is a translation ($d\mu/d\xi=0$): it never enters the Jacobian. For any
invertible $A$ the map is a bijection; “how well $A$ approximates the posterior”
affects sampling efficiency, not correctness of the transformed density.

### 8.3 Conditioner precision contract

| Kind | Meaning |
|---|---|
| Exact diagonal | Live prior precision for a diagonal GP (`gp_block`). |
| CURN inverse-marginal-variance | Elementwise $1/\mathrm{diag}(\Phi)$ reshaped per pulsar for a **dense** global prior (`globalgp_curn_block`). The exact dense prior remains in the likelihood. |
| Caller-declared | `array_block(..., conditioner_precision=...)` — scalar, `(k,)`, or callable with `.params`. |

**No floors / ridges / silent defaults.** Non-finite or negative constant
precisions raise at construction; live callables are checked in `validate`.

### 8.4 Adapters and objects

- `TransportBlock` — name, $F$, one-key localized `index`, conditioner.
- `Transport` — single-pulsar map; true `.params`; `validate`, `diagnostics`,
  `fingerprint`, optional ExtSignal centering and `softclip`.
- `ArrayTransport` — batched equal-width transports (ragged is unsupported).
- `MarginalTransport` — one external block whitened against **live** marginalized
  $C(\eta)$ via `kernel.make_kernelsolve(y_t, W_s)` (no duplicated TNT path).
  Metamath-only. Schema string `discovery-marginal-transport-v1` in fingerprint.
- Reference noise: `reference_noise`, `reference_noise_frozen`; probes
  `reference_noise_quadratic`, `reference_noise_standard_deviation`.
- Live-kernel probes on `MarginalTransport`: `live_kernel_quadratic`,
  `live_kernel_standard_deviation` (Woodbury-stack diagonal walk; projection
  kernels raise).

Cholesky for the transport factor is called **directly** (not through the
configurable LU/Cholesky alias) so an LU config cannot change the transform’s
meaning.

### 8.5 Failure semantics

| Stage | Behaviour |
|---|---|
| Construction | Raise on bad shapes, multi-key blocks, metamath guard, incomplete frozen noise for `decenter=True` sugar |
| `validate(params)` | Eager PD / positivity; returns diagnostics dict on success |
| Runtime `apply` | NaN-propagating under JIT |

---

## 9. Factory and test routes

### 9.1 `_kernels.py`

```text
set_mode("metamath")  → constructors resolve via _METAMATH
set_mode("matrix")    → fall through to matrix.* (monkeypatch still flows)
require_metamath(feature)  → NotImplementedError unless metamath mode
```

### 9.2 Three parity routes (`tests/metamatrix/_routes.py`)

| Route | Likelihood | Kernels |
|---|---|---|
| matrix | `likelihood.py` | matrix |
| mh_patched | `likelihood.py` | metamath via test monkeypatch |
| mh_native | `likelihood_metamath.py` | metamath factory |

The matrix route **must** set mode explicitly and restore the module default —
after the default flip, forgetting this turns every comparison into
self-comparison under metamath.

### 9.3 Autouse fixture

`tests/conftest.py` snapshots/restores kernel mode around each test so the
suite is mode-independent regardless of default.

### 9.4 Coverage

Parity rows exercise, at minimum:

- Single pulsar: white noise, ECORR GP and SM, timing (fixed/variable), power-law
  RN, multi-VGP, fftcov 2D, fixed Fourier variance, delays.
- Global: independent, HD, monopole, compound HD+monopole.
- Array: IRN, combined CRN, HD, decenter, means, CW ExtSignal.

`discovery.recipes` is the shared catalogue with the cookbook.

---

## 10. Summary coefficients column

`summary.py` derives the `coefficients` column from the **assembled kernel’s
`.index`**, not GP type:

| Value | Meaning for coefficient frontend |
|---|---|
| `sampled (k)` | Free coefficients, dimension $k$ |
| `marginalized` | Integrated out (including shadowed under `concat=False`) |
| `projected` | Timing projection path |
| `kernel` | White-noise row |
| `deterministic` | ExtSignal |
| `—` | Residual / delay display rows |

---

## 11. Single precision / reference+delta (internals)

Opt-in only; float64 path unchanged when off.

### 11.1 Design points

1. **Reference is a frozen covariance set**, not merely a parameter point.
   Increment math consumes $\{N_{\mathrm{ref}},\Phi_{\mathrm{ref},k}\}$; a
   $\theta_{\mathrm{ref}}\to$ covariance evaluation is a convenience at the
   likelihood boundary.
2. **Respect fused nesting** — do not flatten the two-level HD Woodbury into a
   single dense level; refdelta twins follow the same nest
   (`vectorwoodburyjointsolve_refdelta` → `globalwoodbury_fused_refdelta`).
3. **Opt-in** — `ArrayLikelihood(reference=...)`; no silent dtype changes.
4. **Timing projection** — `makegp_timing(..., project=True)` is the float32-safe
   improper-prior replacement (`woodbury_proj`).

### 11.2 Half A / Half B

- **Half A:** `combine_f64` / `pin_f64` on final quadratic and log-det assembly.
- **Half B:** resolvent-style increment, not current−reference subtraction of two
  large log$L$s. Outer HD logdet increment reuses Cholesky logdets already paid
  for the solve.

### 11.3 Tests

`tests/single_precision/` — config, graph precision, woodbury/vector/global
refdelta twins, projection, PSD characterisation, reference wiring.

Tutorial: [Single precision](advanced/single_precision).

### 11.4 Design notes

Durable ADRs and research math:

- [Single-precision design](design/single_precision/README)
- [ADR 0001–0004](design/single_precision/adr/0001-reference-is-a-frozen-covariance)
- [Single-level increments](design/single_precision/research_refdelta)
- [Nested / fused increments](design/single_precision/research_nested)
- [Deferred SP work](design/single_precision/future)

---

## 12. Bug ports retained on both routes

These are real fixes, not migration scaffolding:

| Fix | Location |
|---|---|
| `makesampler_nuts` kwargs actually forwarded; unknown names raise | `samplers/numpyro.py` |
| `run_nuts_with_checkpoints` Path mkdir + `to_df` recovery | same |
| `make_uind` empty ECORR basis → empty index table; max epoch size in numpy | `utils.py` (matrix imports it) |
| `WoodburyKernel_varP.make_kernelsolve_simple` | `matrix.py` (metamath already had conditional via graphs) |

---

## 13. Extending the stack (checklist)

1. **New kernel math** → `metamath.py` as `@mm.graph`, return graph; compose leaves.
2. **New factory name** → add to `_kernels._METAMATH` (and matrix fallthrough if
   a matrix class still exists).
3. **New user signal** → `signals.py` / `deterministic.py` via factory + markers.
4. **New likelihood behaviour** → prefer `likelihood_metamath.py`; keep legacy
   parity only if the oracle still needs it.
5. **Parity** → add a recipe (if user-facing) + row in `tests/metamatrix/`.
6. **Transport features** → `transport.py` only; keep kernels graph-pure;
   bake constants at construction; no precision floors.
7. **Docs** → user-facing behaviour in [metamatrix.md](metamatrix); design notes
   here.

---

## 14. Deleting the legacy path (follow-up)

When maintainers sign off after external review, follow the ordered cutover in
[Deletion checklist](design/metamatrix/deletion_checklist) (goldens, drop
monkeypatch route, collapse factory, delete `matrix.py` / legacy
`likelihood.py`, deprecation policy, test/docs cleanup).

Nothing in the current dual-path feature set should depend on those steps having
happened.

Constructor → parity-test map (historical audit, still useful):
[parity_coverage.md](design/metamatrix/parity_coverage).

House rules and graph rationale (design note):
[architecture.md](design/metamatrix/architecture).

---

## 15. Pointers

- User guide: [Metamatrix](metamatrix)
- Design notes index: [design/](design/README)
- Cookbook: [tutorials/cookbook_models](tutorials/cookbook_models)
- API: [api/index](api/index)
- Open multi-ExtSignal cross-term issue: [nanograv/discovery#137](https://github.com/nanograv/discovery/issues/137)
- Source of truth for behaviour: `src/discovery/metamath.py`,
  `likelihood_metamath.py`, `transport.py`, and `tests/metamatrix/`.
