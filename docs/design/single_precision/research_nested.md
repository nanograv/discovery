# Nested reference+delta increment for the fused path (VERIFIED f64)

> **Status (verified, exact).** Checked against an mpmath (60-digit) ground truth
> with the decomposition and the brute-force dense $\Sigma$ built from *identical*
> inputs: the fused decomposition is algebraically **exact** (matches brute $\ln L$ to
> $\sim2\times10^{-14}$, $=$ the f64$\to$mpmath input-conversion residual), and the
> **f64 increment formula holds machine precision ($\sim10^{-16}$ absolute) at every
> move size** ($10^{-1}\ldots10^{-9}$). The three formerly-flagged points in §6 are
> resolved.
>
> The point worth keeping: a naive f64 *difference* of two $\ln L$'s ($\sim10^3$ each,
> dense $\Sigma$ solve at $\mathrm{cond}\sim10^3$) is only good to $\sim10^{-9}$ — a
> $\sim7$-digit loss. The direct increment avoids that subtraction and stays at
> $\sim 10^{-16}$. That 7-digit gap is exactly what destroys the naive path in float32
> and exactly what reference+delta sidesteps.

Companion to `research_note_on_split_with_reference.md`. That note worked out the
single-level increments (§2 quadratic, §3 merged logdet) and the sampled-WN extension
(§7, two perturbations). **This note carries them into the fused two-level path**
(`vectorwoodburyjointsolve` → `globalwoodbury_fused`), where the inner per-pulsar
intrinsic-red-noise (IRN) GP *and* the outer cross-pulsar (GW) GP are both sampled.

The result is mechanical: it is the §2 inner resolvent increment (batched per pulsar),
propagated through the *linear* projection, fed as the $\Delta v$/$\Delta G$ of the §7
outer two-perturbation increment. Batching is preserved because the inner step stays
per-pulsar.

---

## 0. Quantities, matched to the code

Per pulsar $i$, all **fixed** (white noise $N_i$ frozen, bases fixed) → fold to f64
constants (`vectorwoodburyjointsolve`):

| symbol               | code              | definition                                                                                           | shape                                     |
| -------------------- | ----------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| $a_i$                | `ytNmy_consts[i]` | $y_i^{\mathsf T} N_i^{-1} y_i$ (leaf data term, **pinned f64**)                                      | scalar                                    |
| $b_{\mathrm{in},i}$  | `FtNmy_in[i]`     | $F_{\mathrm{in},i}^{\mathsf T} N_i^{-1} y_i$                                                         | $m_{\mathrm{in}}$                         |
| $b_{\mathrm{out},i}$ | `FtNmy_out[i]`    | $F_{\mathrm{out},i}^{\mathsf T} N_i^{-1} y_i$                                                        | $m_{\mathrm{out}}$                        |
| $G_{\mathrm{in},i}$  | `FtNmF_in[i]`     | $F_{\mathrm{in},i}^{\mathsf T} N_i^{-1} F_{\mathrm{in},i}$                                           | $m_{\mathrm{in}}\times m_{\mathrm{in}}$   |
| $G_{\mathrm{out},i}$ | `FtNmF_out[i]`    | $F_{\mathrm{out},i}^{\mathsf T} N_i^{-1} F_{\mathrm{out},i}$                                         | $m_{\mathrm{out}}\times m_{\mathrm{out}}$ |
| $H_i$                | `FtNmF_cross[i]`  | $F_{\mathrm{in},i}^{\mathsf T} N_i^{-1} F_{\mathrm{out},i}$ (`FtNmF_cross_out[i]`$=H_i^{\mathsf T}$) | $m_{\mathrm{in}}\times m_{\mathrm{out}}$  |

**Sampled:** inner prior $\Phi_{\mathrm{in},i}$ (per-pulsar IRN, diagonal for
power-law) and outer prior $\Phi_{\mathrm{gw}}$ (dense across pulsars via the ORF). The
code carries the inverses $P^{\mathrm{m}}_{\mathrm{in},i}=\Phi_{\mathrm{in},i}^{-1}$,
$P^{\mathrm{m}}_{\mathrm{gw}}=\Phi_{\mathrm{gw}}^{-1}$.

**Inner solve** (batched over $i$):

$$
C_{\mathrm{in},i} = \Phi_{\mathrm{in},i}^{-1} + G_{\mathrm{in},i},\qquad
\mu_{y,i} = C_{\mathrm{in},i}^{-1} b_{\mathrm{in},i},\qquad
\mu_{F,i} = C_{\mathrm{in},i}^{-1} H_i .
$$

**Projection** (per pulsar):

$$
\tilde a_i = a_i - b_{\mathrm{in},i}^{\mathsf T}\mu_{y,i},\qquad
\tilde b_i = b_{\mathrm{out},i} - H_i^{\mathsf T}\mu_{y,i},\qquad
\tilde G_i = G_{\mathrm{out},i} - H_i^{\mathsf T}\mu_{F,i},
$$

(i.e. `ytNmy_proj[i]`, `FtNmy_proj[i]`, `FtNmF_proj[i]`).

**Outer solve** (`globalwoodbury_fused`), stacking over pulsars
($\tilde a = \sum_i \tilde a_i$, $\tilde b = \mathrm{stack}_i\,\tilde b_i$,
$\tilde G = \mathrm{blockdiag}_i\,\tilde G_i$):

$$
C_{\mathrm{out}} = \Phi_{\mathrm{gw}}^{-1} + \tilde G,\qquad
\nu = C_{\mathrm{out}}^{-1}\tilde b,\qquad
\ln L = -\tfrac12\big[\,\tilde a - \tilde b^{\mathsf T}\nu + \mathcal L_{\det}\,\big] + \text{const},
$$

where the cross-pulsar coupling enters only through $\Phi_{\mathrm{gw}}^{-1}$ (the dense
ORF block).

**Log-determinant** ($\theta$-dependent part; $\ln\det N_i$ is fixed and cancels). Using
the §3 identity $\ln\det\Phi + \ln\det C = \ln\det(I+\Phi G)$ at each level:

$$
\mathcal L_{\det} = \underbrace{\sum_i \ln\det\!\big(I + \Phi_{\mathrm{in},i} G_{\mathrm{in},i}\big)}_{\text{inner, batched per pulsar}} \;+\; \underbrace{\ln\det\!\big(I + \Phi_{\mathrm{gw}} \tilde G\big)}_{\text{outer}} .
$$

The total GP-explained quadratic is
$Q_{\mathrm{tot}} = \sum_i b_{\mathrm{in},i}^{\mathsf T}\mu_{y,i} + \tilde b^{\mathsf T}\nu$
(inner + outer explained), and $\tilde a - \tilde b^{\mathsf T}\nu = a_{\mathrm{leaf}} - Q_{\mathrm{tot}}$
with $a_{\mathrm{leaf}} = \sum_i a_i$ **fixed** — so the fixed leaf cancels in the
difference, as required.

---

## 1. Reference baselines (computed once, f64)

Supply two reference covariances: $\Phi^{\mathrm{ref}}_{\mathrm{in},i}$ (per pulsar) and
$\Phi^{\mathrm{ref}}_{\mathrm{gw}}$. Evaluate *everything in §0 at the reference* in f64
and store $C^{\mathrm{ref}}_{\mathrm{in},i}$, $\mu^{\mathrm{ref}}_{y,i}$,
$\mu^{\mathrm{ref}}_{F,i}$, $\tilde a^{\mathrm{ref}}_i$, $\tilde b^{\mathrm{ref}}_i$,
$\tilde G^{\mathrm{ref}}_i$, $C^{\mathrm{ref}}_{\mathrm{out}}$,
$\nu_{\mathrm{ref}} = (C^{\mathrm{ref}}_{\mathrm{out}})^{-1}\tilde b^{\mathrm{ref}}$, and
$\ln L_{\mathrm{ref}}$. Under fixed white noise these all fold to f64 constants
automatically.

---

## 2. Inner increments (batched per pulsar) — §2 resolvent

With $\Delta\Phi_{\mathrm{in},i} = \Phi_{\mathrm{in},i} - \Phi^{\mathrm{ref}}_{\mathrm{in},i}$
(diagonal; form in log-space, §5 of the companion) and
$\Delta D_{\mathrm{in},i} = \Phi_{\mathrm{in},i}^{-1} - (\Phi^{\mathrm{ref}}_{\mathrm{in},i})^{-1}$,
the resolvent identity gives

$$
\Delta\mu_{y,i} = \big(C_{\mathrm{in},i}^{-1} - (C^{\mathrm{ref}}_{\mathrm{in},i})^{-1}\big) b_{\mathrm{in},i} = -\,C_{\mathrm{in},i}^{-1}\,\Delta D_{\mathrm{in},i}\,\mu^{\mathrm{ref}}_{y,i},\qquad
\Delta\mu_{F,i} = -\,C_{\mathrm{in},i}^{-1}\,\Delta D_{\mathrm{in},i}\,\mu^{\mathrm{ref}}_{F,i}.
$$

Because $\Phi_{\mathrm{in}}$ is **diagonal**, form $\Delta D_{\mathrm{in},i}$ per mode
($1/\Phi_{\mathrm{in},j} - 1/\Phi^{\mathrm{ref}}_{\mathrm{in},j}$, computed safely as
$-\Delta\Phi_j/(\Phi_j\Phi^{\mathrm{ref}}_j)$ — no inverse-difference cancellation). Both
$\Delta\mu$'s reuse the same $C_{\mathrm{in},i}^{-1}\Delta D_{\mathrm{in},i}$ applied to
the reference $\mu$'s → one batched per-pulsar solve. **Batching preserved.**

---

## 3. Projection increments (per pulsar) — linear, direct

$b_{\mathrm{in},i}, b_{\mathrm{out},i}, H_i$ are fixed, so the projected increments are
linear in the inner $\Delta\mu$:

$$
\Delta\tilde a_i = -\,b_{\mathrm{in},i}^{\mathsf T}\Delta\mu_{y,i},\qquad
\Delta\tilde b_i = -\,H_i^{\mathsf T}\Delta\mu_{y,i},\qquad
\Delta\tilde G_i = -\,H_i^{\mathsf T}\Delta\mu_{F,i}.
$$

No big-minus-big: each is a small contraction of the (small) inner increment. Stack to
$\Delta\tilde b = \mathrm{stack}_i\,\Delta\tilde b_i$,
$\Delta\tilde G = \mathrm{blockdiag}_i\,\Delta\tilde G_i$,
$\Delta\tilde a = \sum_i \Delta\tilde a_i$.

---

## 4. Outer increment — §7 two-perturbation

The outer level sees **two** perturbations:
$\Delta D_{\mathrm{gw}} = \Phi_{\mathrm{gw}}^{-1} - (\Phi^{\mathrm{ref}}_{\mathrm{gw}})^{-1}$
(from the GW prior; route via $\Delta\Phi_{\mathrm{gw}} = \Phi_{\mathrm{gw}} - \Phi^{\mathrm{ref}}_{\mathrm{gw}}$, dense ORF)
and $\Delta\tilde G$ (from the inner GP). Combined:
$\Delta K_{\mathrm{out}} = \Delta D_{\mathrm{gw}} + \Delta\tilde G$. Apply the companion's
§7 forms with $v\to\tilde b$, $C\to C_{\mathrm{out}}$.

**Quadratic** ($\nu = C_{\mathrm{out}}^{-1}\tilde b$,
$\nu_{\mathrm{ref}} = (C^{\mathrm{ref}}_{\mathrm{out}})^{-1}\tilde b^{\mathrm{ref}}$):

$$
\Delta Q_{\mathrm{out}} = \Delta\tilde b^{\mathsf T}\nu \;+\; (\tilde b^{\mathrm{ref}})^{\mathsf T}\,C_{\mathrm{out}}^{-1}\big(\Delta\tilde b - \Delta K_{\mathrm{out}}\,\nu_{\mathrm{ref}}\big).
$$

**Outer logdet** (merged, §3/§7), with
$S_{0,\mathrm{out}} = I + \Phi^{\mathrm{ref}}_{\mathrm{gw}}\tilde G^{\mathrm{ref}}$ and
**current** $\tilde G$ (see §6.1):

$$
\Delta\mathcal L_{\det,\mathrm{out}} = \operatorname{slogdet}\!\Big(I + S_{0,\mathrm{out}}^{-1}\big(\Phi^{\mathrm{ref}}_{\mathrm{gw}}\,\Delta\tilde G + \Delta\Phi_{\mathrm{gw}}\,\tilde G\big)\Big).
$$

**Inner logdet increment** (batched per pulsar, §3, $G_{\mathrm{in},i}$ fixed,
$S_{0,\mathrm{in},i} = I + \Phi^{\mathrm{ref}}_{\mathrm{in},i} G_{\mathrm{in},i}$):

$$
\Delta\mathcal L_{\det,\mathrm{in},i} = \operatorname{slogdet}\!\Big(I + S_{0,\mathrm{in},i}^{-1}\,\Delta\Phi_{\mathrm{in},i}\,G_{\mathrm{in},i}\Big),\qquad
\Delta\mathcal L_{\det} = \sum_i \Delta\mathcal L_{\det,\mathrm{in},i} + \Delta\mathcal L_{\det,\mathrm{out}}.
$$

---

## 5. Assembly

$$
\Delta\ln L = -\tfrac12\big[(\Delta\tilde a - \Delta Q_{\mathrm{out}}) + \Delta\mathcal L_{\det}\big],\qquad
\ln L(\theta) = \underbrace{\ln L_{\mathrm{ref}}}_{\text{f64}} + \underbrace{\Delta\ln L}_{\text{f32}}
$$

(final scalar add in f64). Equivalently $\tfrac12\Delta Q_{\mathrm{tot}} - \tfrac12\Delta\mathcal L_{\det}$
with $\Delta Q_{\mathrm{tot}} = -\Delta\tilde a + \Delta Q_{\mathrm{out}}$. Everything past
the inner batched solve lives in the small $n_{\mathrm{gp}}$ space and is f32-safe.

---

## 6. Verification (done) and the resolved points

A standalone check (now `tests/single_precision/test_refdelta_nested.py`) builds the
nested likelihood two ways and compares the increment to an mpmath (60-digit) ground
truth. **Both the decomposition and the brute-force $\Sigma$ must be built from
identical inputs** — an early version built $\Sigma$ in f64 and the decomposition in
mpmath, which manufactured a spurious $\sim5\times10^{-10}$ "disagreement" that was
really a $10^{-14}$ input mismatch amplified by $\mathrm{cond}\cdot|\ln L|$. Once fixed
(2 pulsars, $m_{\mathrm{in}}=4$, $m_{\mathrm{out}}=3$, $|\ln L|\sim1670$):

```
decomp - brute  (both mpmath, exact identity)         2e-14   <- algebraically exact
move    f64 FORMULA err     f64 BRUTE-difference err
1e-1    3.0e-16             1.4e-9
1e-3    3.6e-17             1.8e-9
1e-6    1.2e-16             1.4e-9
1e-9    1.5e-17             1.9e-9
```

The formula holds $\sim10^{-16}$ regardless of move size; the brute *difference* is
pinned at $\sim10^{-9}$ (its own conditioning, not the formula's). Earlier numbers that
called $\sim10^{-9}$ a "f64 floor" were comparing the exact formula against that
inaccurate brute reference — the formula was never the limit.

Resolutions of the formerly-flagged points:

1. **$\Delta\mathcal L_{\det,\mathrm{out}}$** uses **current** $\tilde G$ in the
   $\Delta\Phi_{\mathrm{gw}}\tilde G$ term. Proof:
   $\Phi_{\mathrm{gw}}\tilde G - \Phi^{\mathrm{ref}}_{\mathrm{gw}}\tilde G^{\mathrm{ref}} = \Phi^{\mathrm{ref}}_{\mathrm{gw}}\Delta\tilde G + \Delta\Phi_{\mathrm{gw}}\tilde G$
   identically, so the boxed §4 form follows with current $\tilde G$.
2. **Sign/convention:** compute $\ln\det(I+\Phi G)$ directly via the §3 identity at each
   level (don't reconstruct from the code's separate `lP`/`lS`; map onto this when
   wiring the graph, but the math object is $\ln\det(I+\Phi G)$).
3. **$\Delta Q_{\mathrm{out}}$ cross term** is the exact resolvent identity (current
   $C_{\mathrm{out}}^{-1}$, $\nu$; reference $\tilde b^{\mathrm{ref}}$, $\nu_{\mathrm{ref}}$)
   — no approximation; collapses to §2/§3 when $\Delta\tilde G = 0$.

Caveat for the test: draw $\Phi_{\mathrm{gw}}$ conditioned (the toy occasionally hit an
ill-conditioned random SPD, harmless here but worth avoiding in CI).
