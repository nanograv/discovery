# Metamatrix: graph-based kernels

*Status of the `metamatrix-meyers` branch. Audience: Discovery developers.*

Source links below point at the `metamatrix-meyers` branch on GitHub.

## TL;DR

`matrix.py` enumerated the Woodbury kernel `Σ = N + FPFᵀ` as a combinatorial
family of hand-written const/var classes. **Metamatrix** replaces that with a
small graph DSL: kernels are expressed once, symbolically, and graph *folding*
specialises constant vs variable inputs at trace time — so the variant explosion
collapses to one class per concept. Both paths are live behind
`ds.config(kernels=…)`, and a suite of tests that build each model both ways and
compare the results certifies the graph path reproduces `matrix.py` to machine
precision — the prerequisite for eventually deprecating `matrix.py`.

This builds directly on Michele's original metamatrix design; the work here
carried it to feature-completeness (Phases 0–4) and added the test scaffolding,
the runtime switch, and the example/cookbook surface.

## Motivation

For `Σ = N + FPFᵀ`, each of `N`, `F`, `P` is independently fixed-at-trace-time
("constant") or parameter-dependent ("variable"). `matrix.py` enumerates the
combinations by hand:

- **Woodbury:** [`WoodburyKernel_novar`, `_varP`, `_varN`, `_varNP`, `_varFP`,
  `VectorWoodburyKernel_varP`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/matrix.py)
  — six classes.
- **NoiseMatrix:** `NoiseMatrix1D_{novar,var}`, `NoiseMatrix2D_{novar,var}`,
  `NoiseMatrixSM_{novar,var}`, `VectorNoiseMatrix{1D,2D}_var` plus the
  `…12D_var` ndim-dispatchers.

Adding a feature meant touching 4–6 sibling classes. The graph rewrite expresses
the math symbolically and lets folding handle const-vs-var, eliminating the
combinatorial surface. See
[`dev_architecture/metamatrix/metamatrix_architecture.md`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/dev_architecture/metamatrix/metamatrix_architecture.md)
for the design rationale and the house rules for writing graph code.

## Architecture / module map

| Module | Role |
|---|---|
| [`metamatrix.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/metamatrix.py) | The graph DSL: leaves, nodes, `@graph`, folding/pruning, and `func()` which materialises a graph into a callable. `func()` is called **once**, at the likelihood boundary. |
| [`metamath.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/metamath.py) | The kernel/GP layer built on the DSL: `NoiseMatrix{,1D,2D,SM}`, `WoodburyKernel`, `GlobalWoodburyKernel`, `VectorWoodburyKernel`, `CompoundGP`, `CompoundDelay`. Methods return graphs; they never call `func()` or materialise internally. |
| [`utils.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/utils.py) | Path-neutral substrate: numerical backend config (`config(backend=…)`, the `jnp`/`jsp`/precision/factorisation globals, `cgsolve`, `make_logdet_estimator`) **and** the GP/Kernel marker types (`Kernel`, `GP`, `ConstantGP`, `VariableGP`, `GlobalVariableGP`, `ExtSignal`) + the indexed Sherman–Morrison helpers. Imported by both paths. |
| [`_kernels.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/_kernels.py) | The kernel-constructor **factory**. `signals.py` builds kernels by name through it; `set_mode()` (driven by `ds.config`) selects matrix vs metamath classes. |
| [`_kernel_switch.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/_kernel_switch.py) | Used only by the tests: temporarily replaces the `matrix.*` kernel names with their metamath equivalents, so the *legacy* likelihood code can be run on the *new* kernels without editing it. It reads its name→class map from `_kernels`, so the two can never disagree. |
| [`signals.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/signals.py) / [`measurement_noise.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/measurement_noise.py) | Model builders (`makegp_*`, `makecommongp_*`, `makeglobalgp_*`, `makedelay`, …). Build kernels through the factory + markers, so they are backend-agnostic. |
| [`likelihood.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/likelihood.py) | The original closure-based likelihoods built on `matrix.py`. Kept as the reference implementation we compare the new path against; removed once `matrix.py` is deprecated. |
| [`likelihood_metamath.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/likelihood_metamath.py) | Metamath-native likelihoods (`PulsarLikelihood`, `GlobalLikelihood`, `ArrayLikelihood`), composing metamath kernels directly. Replaces `likelihood.py` once `matrix.py` is deprecated. |
| [`recipes/`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/recipes/__init__.py) | Importable "model zoo" — the exact models the comparison tests exercise and the [cookbook](tutorials/cookbook_models) renders. |

## Kernel routes and the `config` switch

`ds.config(kernels='matrix'|'metamath')` selects the kernel subsystem (distinct
from `utils.config(backend='numpy'|'jax')`, which selects the numerical backend).
It does two things: sets the `_kernels` factory mode, and rebinds the top-level
`PulsarLikelihood`/`GlobalLikelihood`/`ArrayLikelihood` to the matrix or metamath
implementation. Call it **before** building a model.

`signals.py` never names a kernel implementation — it calls
`kernels.NoiseMatrix1D_var(…)` etc., and the factory resolves the name per mode
(metamath classes from a canonical map, else fall through to `matrix.*`). This
replaced the persistent `matrix.*` monkeypatch that the production metamath path
formerly relied on.

The comparison tests build **every model three ways** and check the results
match:

| How the model is built | What it isolates |
|---|---|
| original likelihood + `matrix.py` kernels | the reference result |
| original likelihood + metamath kernels (swapped in via the test-only monkeypatch) | that the new *kernels* alone reproduce the reference, independent of the likelihood rewrite |
| new metamath likelihood + metamath kernels | that the end-state path reproduces the reference |

The monkeypatch is now used only by these tests; the production metamath path
uses the explicit factory instead.

## Old → new mapping

Graph folding dissolves the const/var (and the 1D/2D/vector) distinctions, so the
matrix variant families collapse to one class each.

| `matrix.py` (old) | metamath (new) | Note |
|---|---|---|
| `WoodburyKernel_{novar,varP,varN,varNP,varFP}` | [`WoodburyKernel`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/metamath.py) | const/var by folding |
| `VectorWoodburyKernel_varP` | `VectorWoodburyKernel` | + `make_conditional` (new; matrix had none) |
| — (assembled ad hoc) | `GlobalWoodburyKernel` | first-class global (HD) kernel |
| `NoiseMatrix1D_{novar,var}` | `NoiseMatrix1D` | array *or* callable |
| `NoiseMatrix2D_{novar,var}` | `NoiseMatrix2D` | array *or* callable |
| `NoiseMatrix12D_var` (dispatch) | `NoiseMatrix12D` | ndim dispatch retained |
| `NoiseMatrixSM_{novar,var}` | `NoiseMatrixSM` | indexed Sherman–Morrison |
| `VectorNoiseMatrix{1D,2D,12D}_var` | `NoiseMatrix1D/2D/12D` | "vector" distinction dissolved |
| `CompoundGP`, `VectorCompoundGP` | `CompoundGP` | + mixed-Φ marginalised path (new) |
| `CompoundGlobalGP` (in `matrix.py`) | `signals.CompoundGlobalGP` | relocated, backend-agnostic |
| `CompoundDelay` | `CompoundDelay` | unchanged contract |

The factory's name→class map is the canonical encoding of this table:
[`_kernels._METAMATH`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/_kernels.py).

## Functionality: changed and added

- **Measurement-noise collapse.** `makenoise_measurement{,_simple}` moved to
  [`measurement_noise.py`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/measurement_noise.py)
  in collapsed form: the four `_novar`/`_var` terminal returns became two
  variant-agnostic factory calls (`NoiseMatrix1D` / `NoiseMatrixSM`), the variant
  chosen from whether the argument is an array or a callable. `signals.py`
  re-exports them.
- **Mixed-Φ `CompoundGP`.** A compound of a per-pulsar-diagonal (1D) GP and a
  dense (2D, e.g. HD) GP. The coefficient-log-prior / `Phi=None` branch is now
  gated to the vector/decentered path; a mixed but *marginalised* compound builds
  a real combined dense Φ (block-diagonal, 1D blocks promoted to dense).
- **`CompoundGlobalGP` relocated** out of `matrix.py` to a backend-agnostic
  `signals.CompoundGlobalGP` (factory + markers), so a list of global GPs
  (e.g. HD + monopole) works under both paths.
- **ExtSignal / continuous wave.** `ExtSignal` (in `utils.py`) carries a
  deterministic signal on its own higher-frequency Fourier basis;
  [`makecw_extsignal`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/deterministic.py)
  + `make_extsignal_fourier` fold it into `ArrayLikelihood` via GP cross-terms,
  with the CW parameters never entering the GP prior.
- **Decentering, means, conditional.** `ArrayLikelihood` supports decentered
  (whitened-coefficient) sampling, per-GP prior `means`, and `conditional` /
  `clogL` on the vector Woodbury path (the latter has no `matrix.py` analogue).
- **2-D covariance bases.** `makegp_fftcov`/`avgcov`/`intcov` and
  `makegp_fourier_variance` (dense `NoiseMatrix2D`) run through the graph path.
- **Backend substrate.** Numerical config and the GP/Kernel markers were pulled
  out of `matrix.py` into `utils.py`, so neither path reaches into the other.

## What has been checked

The comparison tests (`tests/metamatrix/`) check that `logL`, `conditional`,
`clogL`, `sample`, and `sample_conditional` agree — between the original
`matrix.py` path and the new metamath path (both run through the original
likelihood and through the new metamath likelihood) — to machine precision, on
real NANOGrav pulsars. Coverage spans, at minimum:

- **Single pulsar:** white noise (simple + per-backend), ECORR as a GP and folded
  via Sherman–Morrison, marginalised + variable timing models, power-law red
  noise (incl. `concat=False` chained Woodburys), multiple variable GPs (RN + DM),
  2-D `fftcov` and fixed-variance Fourier GPs, deterministic delays.
- **Global:** independent pulsars, HD- and monopole-correlated global GPs, and a
  compound (HD + monopole) global GP.
- **Array:** intrinsic (vectorised) red noise, intrinsic + common-spectrum via
  `make_combined_crn`, intrinsic + HD global, decentered variants, per-GP means,
  and a CW ExtSignal.

Every kernel constructor that production code or the example notebooks emit is
exercised by at least one of these models; the audit and disposition of each is in
[`dev_architecture/metamatrix/phase3_coverage.md`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/dev_architecture/metamatrix/phase3_coverage.md).
These same models are the importable [`discovery.recipes`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/src/discovery/recipes/__init__.py)
rendered in the [cookbook](tutorials/cookbook_models) — so the documented models
and the checked models are the same objects.

A real-scale check (5 pulsars, CW + HD global + decentered common RN) agrees
between the two kernel paths to ~1e-9 relative.

### Not yet cross-checked

- **`cglogL`** — the conjugate-gradient log-likelihood for very large arrays
  (CG solve + stochastic-Lanczos log-det estimator) is implemented on the
  metamath-native path but is **not** in the comparison tests; neither path's
  `cglogL` is currently cross-checked.
- **Performance.** The comparison-test suite checks correctness, not speed — there
  is no general `matrix.py`-vs-metamath timing comparison. (The single-precision
  work *does* have an HD-model benchmark: matrix vs metamath, `float64` vs `float32`;
  see the section below.)
- **GPU.** The comparison tests run on CPU in `float64`. The graph path is
  JAX-native; it has now been exercised on GPU (GH200) for the single-precision HD
  benchmark below, but broader GPU coverage of the comparison suite is still pending.
- **Samplers.** The numpyro samplers ride on `logL`/`clogL` (which are checked),
  but no end-to-end sampling comparison has been run.

## Single precision (float32) on GPU

GPUs are far faster in `float32`, but a PTA marginal `logL` subtracts large numbers
(a quadratic form $\sim 10^6$, log-determinants $\sim 10^4$) to land on an $O(1)$
result. At the `float32` ulp of a $10^6$-scale number ($\approx 0.06$) that result is
swamped, so a blanket-`float32` likelihood is too imprecise to sample with. Three
complementary pieces address this. All are **opt-in** and byte-identical to the
`float64` path when off; the user-facing comparison is in the
[single-precision tutorial](advanced/single_precision).

- **Half A — `float64` final combine.** The expensive factorisations stay in the
  working (`float32`) dtype, but the final scalar assembly (the big `ytNmy − quad`
  subtraction and the log-determinant sum) is done in `float64` (`combine_f64` /
  `pin_f64`), with the per-pulsar raw quadratics and white-noise log-dets pinned to
  `float64` at source. This protects the *combination*, not the building of `mu`
  (which is still born from the `float32` Cholesky).
- **Half B — reference + delta.** Freeze each GP level's prior covariance once at a
  reference point $\theta_{\rm ref}$ (in `float64`), and evaluate
  $\ln L(\theta) = \ln L_{\rm ref} + \Delta\!\ln L$: $\ln L_{\rm ref}$ carries the large
  pieces (`float64`, computed once) and `float32` only ever holds the $O(1)$
  increment, built directly via a resolvent identity (never a current−reference
  subtraction of two large log-likelihoods). Opt in with
  `ArrayLikelihood(..., reference=θ_ref)`. The fused two-level HD path has
  reference+delta twins (`vectorwoodburyjointsolve_refdelta` →
  `globalwoodbury_fused_refdelta`).
- **Timing-model projection.** Handle the timing model $M$ by projection (the exact
  flat-prior limit) instead of a huge-variance prior — `float32`-safe, and it keeps
  the corrections. Opt in with `makegp_timing(..., project=True)`.

Status (12 pulsars, NG15 HD model; precision vs the `float64` truth):

| flavor | max \|ΔlogL\| vs truth | note |
|---|---|---|
| `float64` | $\sim 10^{-8}$ | machine precision |
| blanket `float32` | $\sim 7\times 10^{-2}$ | sampling-fatal; worsens with array size |
| reference+delta `float32` | $\sim 10^{-3}$ (CPU) | on GPU the floor is TF32 ($\sim 10^{-2}$) |

On speed, the reference+delta fused HD kernel now costs **~1.5× the `float64` path's
FLOPs** (and ~1.5× its wall-clock on a GH200), down from ~74× before the outer
log-determinant increment was rewritten to reuse the Cholesky log-dets it already
computes. So the accurate `float32` path is roughly as fast as the normal path while
recovering most of the precision.

**Done:** Half A, Half B (incl. the fused HD reference+delta twins and the
outer-increment speedup), and projection are implemented and have a precision/speed
harness + unit tests. **Remaining:** scaling re-measurement at 45/67 pulsars on GPU;
optionally recovering the GPU TF32 floor with `jax_default_matmul_precision='highest'`;
and routing reference+delta for the single-level (CURN/IRN) `vectorwoodbury` path
(currently HD-only). Design decisions and notes live in
[`dev_architecture/single_precision/`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/dev_architecture/single_precision/)
(README, research notes, and ADRs 0001–0004).

## Status and remaining work

- **The metamath path is feature-complete.** Shared substrate extracted,
  `signals.py` building through the factory, `likelihood_metamath` free of
  `matrix.py`, every used constructor checked, and the last two hard cases closed
  (the all-constant 2-D GP prior, and combining a list of global GPs). Nothing in
  the comparison tests is left failing or skipped.
- **What remains is the deprecation itself:** once we're confident enough to drop
  `matrix.py`, we delete it and the original `likelihood.py`, collapse the factory
  to metamath-only, drop the test monkeypatch, and rename `likelihood_metamath.py`
  → `likelihood.py`. That's deliberately held until the metamath path has been
  exercised on real analyses (this review is part of that).

The step-by-step plan and the conditions for each step are in
[`dev_architecture/metamatrix/exit_plan.md`](https://github.com/meyers-academic/discovery/blob/metamatrix-meyers/dev_architecture/metamatrix/exit_plan.md).

## Pointers

- [Model cookbook](tutorials/cookbook_models) — runnable gallery of every tested model.
- [API reference](api/index) — `[source]` links to each function/class.
- `dev_architecture/metamatrix/` — design notes, exit plan, coverage audit.
