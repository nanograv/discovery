# Future work — single precision

Deferred ideas. Not started. Each entry: what, why, and a rough how.
Product docs: [Metamatrix user guide](../../metamatrix.md),
[developer guide](../../metamatrix_dev.md), [tutorial](../../advanced/single_precision).

## 1. Configurable pin set

**What.** Today the float64 pins are hardcoded in `metamath.woodbury`
(`g.pin_f64(ytNmy)`, `g.pin_f64(lN)`, `g.pin_f64(lP)`). The set of pinned
quantities is fixed in the kernel math. Make *which* quantities are pinned a
config choice, without editing the kernel math.

**Why.**
- The design (see the single-precision design README) left an explicit open question: which of
  `{ytNmy, lN, lP, lS, ld}` should be pins. Confirming that wants experiments on
  different datasets, not code edits.
- Lets us run accuracy/perf sweeps: is `lP` ever load-bearing? does pinning `lS`
  (and paying for a float64 Cholesky) ever actually help on a big array?

**How (sketch).** Move from a boolean to a *label*, and split intent (in the
math) from policy (in config):
- `Node.pin` becomes `Optional[str]` (a label) instead of `bool`.
- `GraphBuilder.pin_f64(sym, key="ytNmy")` tags a node with its label. Still pure
  graph intent — the kernel math advertises the *candidates*; it does not decide
  which are active. House rule (methods return graphs) preserved.
- The policy lives in config and is read at materialization (`func()` boundary),
  in `_dtype_map`:
  ```python
  active = utils.active_pins()          # default: all known labels
  pinned = [name for name, node in graph.items()
            if getattr(node, "pin", None) in active]
  ```
- `utils.config(pins="all" | None | {"ytNmy", "lN"})` selects the set. Default
  "all" = exactly today's behavior, so it's a pure superset and the existing
  tests stay green.
- Add a test: a disabled pin really drops that node to the working dtype (the
  dtype map no longer marks it / its cone float64).

**Cost.** ~3 small touchpoints (`Node`, `_dtype_map`, `pin_f64` signature) + the
config knob + one test. Default-preserving.

**Pairs with** the graph visualization (below): once pins have labels, the
visualization can show *which* pin each float64 region traces back to.

## 2. Unify the two paths — make the reference "impotent" when none is supplied

**What.** Today every refdelta kernel has a *twin*: `vectorwoodbury` vs
`vectorwoodbury_refdelta`, `globalwoodbury_fused` vs
`globalwoodbury_fused_refdelta`, and the kernels branch on whether a frozen
`P_ref` is present (ADR 0003 opt-in). That is two code paths to maintain.

**Why.** When `reference=None` there is no reason we cannot just construct a
reference covariance that is *impotent* — i.e. set Φ_ref ≡ Φ (the live prior) so
every increment is identically zero and logL = logL_ref exactly. If the
no-reference case is expressible as "refdelta with a self-cancelling reference",
the **non-refdelta paths can be deleted**: one kernel per level instead of two.

**How (sketch).**
- When no reference is supplied, feed the refdelta graph `Pinv_ref = Pinv` (the
  *same* live leaf, not a frozen constant). Then ΔΦ = Φ − Φ_ref = 0, ΔD = 0, all
  Δ-quantities vanish, and `logL_ref` carries the whole answer.
- Care needed: with Φ_ref = Φ live, `logL_ref` no longer folds to an f64 constant
  (it now depends on params), so the f64/f32 dtype intent has to be re-checked —
  the reference solve must run at working dtype in that mode, and we lose the
  "reference folds to f64" trick. So this is a *code-simplification* refactor, not
  a free precision win; verify it stays byte-identical to today's non-refdelta path
  (the parity tests in `tests/metamatrix/` are the guard).
- If the dtype story gets awkward, an alternative is keeping one kernel that
  *internally* skips the increment branch when ref is the live leaf — still one
  function, one set of tests.

**Cost.** A focused refactor of the four kernels + their routing in
`likelihood_metamath.py`; the payoff is roughly halving the kernel surface area.

## 3. Pin the outer increment-assembly to f64 (tighten the HD floor ~3 decades)

**What.** Prior fused-HD refdelta experiments put the f32 floor near ~1e-2,
while the increment itself is only ~10 (≈6 orders below |logL|), so the ideal
floor is ~f32-eps×10 ≈ 1e-5. The ~1000× gap is f32 roundoff in the *outer*
increment-assembly matrix ops, whose operands span a huge dynamic range
(GW-block Φ⁻¹ ~ 1e10, covariances ~1e-12).

**Why.** The outer GW block is small (≤ npsr·14, ~630×630 at 45 psr), so pinning
its increment assembly to f64 is nearly free, and there are ~3 decades on the
table. The big *inner* batched Cholesky stays f32 (that is the speed win — must
not be pinned).

**How (sketch).** In `globalwoodbury_fused_refdelta`, `combine_f64`/`pin_f64` the
outer-only nodes: `dD_gw = −Pm·ΔΦ_gw·Pmr`, the dense outer `inv`/`ΔΦ_gw`, and the
outer cross-term `cho_solve(cf, dbt − dK·nu_ref)`. Re-measure against the
float64 fused path (see `tests/single_precision/`); pairs naturally with item 1
(configurable pin set) so the outer-pin policy is a knob, not hardcoded.

## (add future items below as they come up)
