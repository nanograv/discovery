# Single-precision design notes

Durable design for float32-safe metamath likelihoods. **Not** a living project
board: product usage is in the
[single-precision tutorial](../../advanced/single_precision) and the
[user](../../metamatrix.md) / [developer](../../metamatrix_dev.md) metamatrix
guides. Official tests: `tests/single_precision/`.

## Why this exists

PTA marginal log-likelihoods subtract large numbers (quadratic forms and log
dets of size $\sim 10^{4}$–$10^{6}$) to get an $O(1)$ result. Blanket float32
is often too imprecise to sample. Discovery addresses this with **opt-in**
pieces that leave the default float64 path unchanged:

| Piece | Idea |
|---|---|
| **Half A — pins / f64 combine** | Expensive work may run in the working dtype; final scalar assembly uses `pin_f64` / `combine_f64`. |
| **Half B — reference + delta** | Freeze GP prior covariances at a reference; evaluate $\ln L = \ln L_{\mathrm{ref}} + \Delta\ln L$ so float32 holds an $O(1)$ increment. |
| **Timing projection** | `makegp_timing(..., project=True)` — exact flat-prior limit instead of a $10^{40}$ improper variance (overflows float32). |

Scope is the **metamath** graph path only. Graph folding decides const vs live
at trace time, so precision is a materialization rule, not a fixed const/var
class seam as in `matrix.py`.

## Glossary

**Reference covariances ($N_{\mathrm{ref}}$, $\Phi_{\mathrm{ref}}$)**  
Frozen covariances the likelihood expands around (fed as constant graph leaves).
A parameter point $\theta_{\mathrm{ref}}$ is only one way to produce them.
See [ADR 0001](adr/0001-reference-is-a-frozen-covariance.md).

**Increment ($\Delta\ln L$)**  
The small change $\ln L(\theta)-\ln L_{\mathrm{ref}}$, formed analytically —
never the float32 difference of two large totals.

**Pin**  
A graph node built in float64 while the rest of the graph uses the working
dtype. Protects *building* a quantity, not necessarily its later combination.

**Leaf data term**  
The white-noise quadratic $y^{\mathsf T}N_0^{-1}y$ at the bottom of the Woodbury
recursion. Static under fixed white noise; the natural f64 pin.

**Recursive (fused) Woodbury**  
Nested Woodbury: per-pulsar GPs marginalized into effective noise that an outer
(cross-pulsar / HD) GP sees. Flattening into one giant block-$\Phi$ is rejected
([ADR 0002](adr/0002-respect-fused-nesting-no-flattening.md)).

## Architecture decisions (ADRs)

| ADR | Decision |
|---|---|
| [0001](adr/0001-reference-is-a-frozen-covariance.md) | Reference is a frozen covariance set, not a parameter point |
| [0002](adr/0002-respect-fused-nesting-no-flattening.md) | Respect fused nesting; no flattening |
| [0003](adr/0003-reference-delta-is-opt-in.md) | Reference+delta is opt-in (`ArrayLikelihood(reference=...)`) |
| [0004](adr/0004-timing-model-projection.md) | Timing projection = exact flat-prior marginalization |

## Research notes (math)

| Note | Content |
|---|---|
| [research_refdelta.md](research_refdelta.md) | Single-level reference+delta analytic increments |
| [research_nested.md](research_nested.md) | Nested / fused HD increment (verified exact in f64) |

## Deferred work

[future.md](future.md) — e.g. configurable pin sets; not started.

## Implementation map

| Concern | Where |
|---|---|
| Graphs (`woodbury`, `*_refdelta`, `woodbury_proj`, fused HD) | `src/discovery/metamath.py` |
| `reference=` wiring | `src/discovery/likelihood_metamath.py` |
| `makegp_timing(..., project=True)` | `src/discovery/signals.py` |
| Tests | `tests/single_precision/` |
| Tutorial | `docs/advanced/single_precision.ipynb` |
