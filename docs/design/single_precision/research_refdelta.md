# Reference + delta likelihood: the analytic increments (Sec. IV–V distilled)

Analytic increment formulas for the single-level reference+delta likelihood.
Self-contained; companion nested/fused math is in
[research_nested.md](research_nested.md).

---

## 0. Notation (brief notation, used throughout)

Single-pulsar marginalized GP likelihood, Woodbury form:

$$
C(\theta) = \Phi(\theta)^{-1} + F^{\mathsf T}N^{-1}F,
$$
$$
\ln L(\theta) = -\tfrac12\Big[\, y^{\mathsf T}N^{-1}y \;-\; v^{\mathsf T}C(\theta)^{-1}v \;+\; \ln\det N \;+\; \ln\det\Phi(\theta) \;+\; \ln\det C(\theta) \;+\; n\ln 2\pi \,\Big],
$$

with the brief's shorthands

$$
v \equiv F^{\mathsf T}N^{-1}y \;(=\texttt{FtNmy}),\quad
G \equiv F^{\mathsf T}N^{-1}F \;(=\texttt{FtNmF}),\quad
\texttt{ytNmy}\equiv y^{\mathsf T}N^{-1}y .
$$

Sizes: $y\in\mathbb R^{n_{\rm toa}}$ ($n_{\rm toa}\sim10^4$), $F\in\mathbb R^{n_{\rm toa}\times n_{\rm gp}}$, everything with a $C$, $\Phi$, $G$, $v$ lives in the small $n_{\rm gp}$ space ($n_{\rm gp}\sim10^2$). **Fixed white noise** ($N$ frozen from single-pulsar analysis) means $v$, $G$, `ytNmy`, $\ln\det N$ are all $\theta$-independent; only $\Phi(\theta)$ (hence $C$) moves.

**$\Phi$ is a general symmetric positive-definite matrix, not assumed diagonal.** It is diagonal for the per-bin PSD models (power-law, free-spectrum), but the code must also accept dense $\Phi$: FFT-based covariances (Crisostomi et al.) are dense *within* a pulsar, and a full-array Hellings–Downs $\Phi$ is dense *across* pulsars (block-diagonal in frequency, dense $n_p\times n_p$ ORF block per bin). The increment formulas below are written for general $\Phi$; the diagonal case is noted only where it collapses an $m\times m$ operation to a per-mode one. Write $D(\theta)\equiv\Phi(\theta)^{-1}$ for the inverse — a dense matrix in general — used only as an intermediate, never formed by subtracting two inverses (see §2).

The one quantity that maps onto the brief's dangerous term is
$$
\underbrace{v^{\mathsf T}C^{-1}v}_{\text{brief }\texttt{FtNmy}^{\mathsf T}\mu} \;=\; v^{\mathsf T}\mu,\qquad \mu \equiv C^{-1}v .
$$

> Cross-reference for your own notes only (Code does not need it). In the set of notes the symbols are $\delta t = y$, $T = F$, $\mathcal B = \Phi$, $\Sigma = \mathcal B^{-1}+T^{\mathsf T}N^{-1}T = C$, and $\mathcal Q = v^{\mathsf T}C^{-1}v$. The notes' $\mathcal C = N + F\Phi F^{\mathsf T}$ is the *big* covariance, not the small $C$ above — don't conflate them.

---

## 1. The decomposition

$$
\boxed{\;\ln L(\theta) = \ln L(\theta_{\rm ref}) + \Delta\ln L(\theta;\theta_{\rm ref})\;}
$$

with the baseline computed **once in f64** and only $\Delta\ln L$ per call. Splitting the bracket above into its $\theta$-dependent pieces,

$$
\Delta\ln L \;=\; \tfrac12\,\Delta\mathcal Q \;-\; \tfrac12\,\Delta\mathcal L_{\det},
$$
$$
\Delta\mathcal Q \equiv v^{\mathsf T}C(\theta)^{-1}v - v^{\mathsf T}C(\theta_{\rm ref})^{-1}v,
\qquad
\Delta\mathcal L_{\det} \equiv \big[\ln\det\Phi + \ln\det C\big]_\theta - \big[\ln\det\Phi + \ln\det C\big]_{\theta_{\rm ref}} .
$$

Note `ytNmy`, $\ln\det N$, $n\ln2\pi$ cancel exactly in the difference under fixed WN — they never enter $\Delta\ln L$. (This is why the brief keeps `ytNmy` in f64 and does **not** split it.)

The two increments below are each built as a **direct analytic object**, never as the f32 difference of two $\sim\!10^6$ totals.

---

## 2. Quadratic increment (brief item 1)

Resolvent identity (exact, **no diagonal assumption**). With $\Delta D \equiv D(\theta)-D(\theta_{\rm ref}) = \Phi^{-1}-\Phi_{\rm ref}^{-1}$,

$$
C(\theta)^{-1} - C(\theta_{\rm ref})^{-1} \;=\; -\,C(\theta)^{-1}\,\Delta D\,C(\theta_{\rm ref})^{-1},
$$
$$
\Delta\mathcal Q = v^{\mathsf T}\big(C^{-1}-C_{\rm ref}^{-1}\big)v = -\,u^{\mathsf T}\,\Delta D\,u_{\rm ref},
\qquad u\equiv C(\theta)^{-1}v,\;\; u_{\rm ref}\equiv C(\theta_{\rm ref})^{-1}v .
$$

For dense $\Phi$, do **not** form $\Delta D = \Phi^{-1}-\Phi_{\rm ref}^{-1}$ by subtracting two inverses (cancellation, and it needs $\Phi^{-1}$ explicitly). Route it through the resolvent *for $\Phi$ itself*, which moves the difference onto the natural covariance-space increment $\Delta\Phi \equiv \Phi-\Phi_{\rm ref}$:

$$
\Delta D = \Phi^{-1}-\Phi_{\rm ref}^{-1} = -\,\Phi^{-1}\,\Delta\Phi\,\Phi_{\rm ref}^{-1}.
$$

Substituting and folding the $\Phi^{-1}$ factors into the solves gives the clean general form

$$
\boxed{\;\Delta\mathcal Q = w^{\mathsf T}\,\Delta\Phi\,w_{\rm ref},\qquad
w \equiv (I+G\Phi)^{-1}v \;\;(\text{per call}),\quad
w_{\rm ref}\equiv (I+G\Phi_{\rm ref})^{-1}v \;\;(\text{precomputed, f64}).\;}
$$

Equivalently $w=\Phi^{-1}u$ and $w_{\rm ref}=\Phi_{\rm ref}^{-1}u_{\rm ref}$, so if you already form $C$ and its Cholesky you can get $w$ by an extra $\Phi$-solve; otherwise solve the $m\times m$ system $(I+G\Phi)w=v$ directly (an LU, no explicit $\Phi^{-1}$ and no $C$ needed). $\Delta\Phi$ is the genuinely small object — compute it from the model (analytically where possible), since $\Phi$ entries are $\mathcal O(\rho)$, so this covariance-space difference is benign, *not* a $10^6$ cancellation.

- **Diagonal special case** ($\Phi$ per-bin): $\Delta\Phi\to\operatorname{diag}(\Phi_j-\Phi_{{\rm ref},j})$ and the boxed contraction collapses to a per-mode sum $\Delta\mathcal Q=\sum_j (\Phi_j-\Phi_{{\rm ref},j})\,w_j\,w_{{\rm ref},j}$; equivalently $-\sum_j(1/\Phi_j-1/\Phi_{{\rm ref},j})u_j u_{{\rm ref},j}$. Form the diagonal $\Delta\Phi$ in log-space (§5) to dodge under/overflow.
- **General case:** $w^{\mathsf T}\Delta\Phi\,w_{\rm ref}$ is an $m\times m$ quadratic form — still entirely in the small $n_{\rm gp}$ space, $\mathcal O(m^2)$ flops, result $\mathcal O(\sqrt m)$. Use an f64 accumulator for the contraction if you want it bulletproof; cheap.

This is the central win, and it survives non-diagonal $\Phi$ intact: the dangerous $\sim\!10^6$ term is replaced by an $\mathcal O(1)$ object in the $n_{\rm gp}$ space. $w_{\rm ref}$ is born from the f64 reference factorization (done once); $w$ uses the per-call f32 factorization — the one the brief wants kept in f32.

---

## 3. Log-determinant increment (brief items 2 **and** 3, merged)

**Correction to the plan.** The brief lists `logdetC` and `logdetΦ` increments
separately. Do **not** compute them separately: individually each is large and of
opposite sign (each $\sim\!10^3$–$10^5$, see the conservative bound below), and
their difference is a catastrophic cancellation — the same disease, relocated.
Only their **sum** is the well-behaved $\theta$-dependent quantity. Compute it as a
single object. (This holds for general $\Phi$; nothing here needs $\Phi$ diagonal.)

Use the matrix-determinant identity. Since $C=\Phi^{-1}+G$,

$$
\ln\det\Phi + \ln\det C = \ln\det\!\big[\Phi(\Phi^{-1}+G)\big] = \ln\det\!\big(I + \Phi G\big).
$$

So the combined increment is a ratio of two such determinants. With the reference
matrix $S_0 \equiv I + \Phi_{\rm ref}G = \Phi_{\rm ref}C(\theta_{\rm ref})$ (factorized once, f64) and $\Delta\Phi \equiv \Phi-\Phi_{\rm ref}$,

$$
I + \Phi G = S_0\big[\,I + S_0^{-1}\,\Delta\Phi\,G\,\big]
\;\Longrightarrow\;
\boxed{\;\Delta\mathcal L_{\det} = \ln\det\!\big(I + S_0^{-1}\,\Delta\Phi\,G\big)\;}
$$

A single $n_{\rm gp}\times n_{\rm gp}$ log-determinant per call, $\mathcal O(1)$ in magnitude (its exponential is $\det\approx1$, well-conditioned), safe in f32. $\Delta\Phi$ enters only through this small product, so a dense $\Delta\Phi$ (fftcov, HD) changes nothing structurally — it just makes the middle factor a full $m\times m$ rather than a diagonal.

**Reuse the existing factor.** Because $S_0 = \Phi_{\rm ref}\,C(\theta_{\rm ref})$, you do not need a separate factorization of $S_0$: with $L_0L_0^{\mathsf T}=C(\theta_{\rm ref})$ (the same f64 Cholesky from §2) and a $\Phi_{\rm ref}$-solve,

$$
M \equiv S_0^{-1}\Delta\Phi\,G = C(\theta_{\rm ref})^{-1}\,\Phi_{\rm ref}^{-1}\,\Delta\Phi\,G
= L_0^{-\mathsf T}L_0^{-1}\big[\,\Phi_{\rm ref}^{-1}\Delta\Phi\,G\,\big],
\qquad
\Delta\mathcal L_{\det} = \operatorname{slogdet}(I+M).
$$

For diagonal $\Phi$, $\Phi_{\rm ref}^{-1}\Delta\Phi = \operatorname{diag}(\Phi_j/\Phi_{{\rm ref},j}-1)$; for dense $\Phi$, apply the $\Phi_{\rm ref}$ factorization as a solve. $I+M$ is generally non-symmetric; its eigenvalues are positive real (it is similar to a product of SPD operators), so `slogdet` returns sign $+1$ and you take the log-magnitude. An LU in f32 is fine here because $I+M$ is well-conditioned. Build $\Delta\Phi$ from the model directly (Sec. 5 for the diagonal PSD case), not by subtracting two raw $\Phi$'s where that would cancel.

Why the conservative bound matters: a worst-case DM-dominated single pulsar gives $\ln\det(I+\Phi G)=\sum_j 2\ln(1+\lambda_j)\lesssim7\times10^5$ across the array (with $\lambda_j\approx \tfrac{n_k}{2}\,\sigma^2_{j,\rm GP}/\sigma^2_{j,\rm WN}$). That is right at the f32 integer ceiling — which is exactly why you must never materialize `logdetΦ` or `logdetC` alone in f32, and why the increment form above (which never forms either) is the safe route.

---

## 4. What to precompute and store at $\theta_{\rm ref}$ (all f64, once)

| Quantity                       | Definition                                                                                                                                      | Used by                                                      |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| $L_0$                          | Cholesky of $C(\theta_{\rm ref}) = \Phi_{\rm ref}^{-1}+G$                                                                                       | §2 ($w_{\rm ref}$ via $u_{\rm ref}$), §3 ($M$)               |
| $\Phi_{\rm ref}$ factorization | Cholesky/LU of the reference $\Phi_{\rm ref}$ (or its inverse, if diagonal)                                                                     | §2, §3 ($\Phi_{\rm ref}$-solves)                             |
| $w_{\rm ref}$                  | $(I+G\Phi_{\rm ref})^{-1}v = \Phi_{\rm ref}^{-1}C(\theta_{\rm ref})^{-1}v$                                                                      | §2                                                           |
| $v,\;G$                        | $F^{\mathsf T}N^{-1}y,\;F^{\mathsf T}N^{-1}F$                                                                                                   | §2, §3 (θ-indep under fixed WN; **folded to f64 constants**) |
| $\Phi_{\rm ref}$               | reference GP prior covariance (dense or diagonal)                                                                                               | §2, §3, §5                                                   |
| baseline scalars               | `ytNmy`, $\ln\det N$, $v^{\mathsf T}\mu_{\rm ref}=v^{\mathsf T}u_{\rm ref}$, $\ln\det\Phi_{\rm ref}$, $\ln\det C(\theta_{\rm ref})$, $n\ln2\pi$ | assemble $\ln L(\theta_{\rm ref})$                           |

$\ln L(\theta_{\rm ref})$ itself is a stored f64 constant. Per call you compute only the two boxed increments, in f32, on top of it.

---

## 5. PSD evaluation and underflow — diagonal-$\Phi$ models

This section is **specific to per-bin (diagonal) $\Phi$** — power-law and free-spectrum models, where $\Phi_j$ is a scalar power law $\propto A^2 (f_j/f_{\rm yr})^{-\gamma}$ spanning many orders of magnitude. There, evaluate $\ln\Phi_j$ linearly first, then exponentiate, and:

- form $\Delta\Phi_j = \Phi_j - \Phi_{{\rm ref},j}$ and the ratio $\Phi_j/\Phi_{{\rm ref},j}=\exp(\ln\Phi_j-\ln\Phi_{{\rm ref},j})$ from the log-space difference (this feeds $\Delta\Phi$ in §2 and §3 without cancellation);
- **drop underflowing modes:** if $\Phi_j$ falls below the f32 floor ($\sim3.4\times10^{-38}$) its eigenvalue $\lambda_j\to0$, contributes nothing to either increment, and can be masked out. This is a per-mode mask, not an approximation.

**Dense $\Phi$ (fftcov within a pulsar, full-array HD across pulsars).** The §2–§3 increment formulas are unchanged — they only ever see $\Delta\Phi=\Phi-\Phi_{\rm ref}$ and $\Phi_{\rm ref}$-solves. What changes is how $\Phi$ and $\Delta\Phi$ are *built*, and that is model-specific:

- **HD full-array:** $\Phi$ is block-diagonal in frequency with a dense $n_p\times n_p$ ORF block $\Gamma\,S(f_k)$ per bin. $\Delta\Phi$ is then block-wise $\Gamma\,\big(S(f_k)-S_{\rm ref}(f_k)\big)$ — compute the scalar spectrum increment in log-space as above, then scale the fixed $\Gamma$ block. Cheap and analytic.
- **fftcov:** $\Phi$ is dense within a pulsar; build $\Delta\Phi$ from the model's own parametrization. No per-element log-space trick applies to off-diagonal entries, but the covariance-space difference is benign (entries are $\mathcal O(\rho)$, not $10^6$), so a direct dense subtraction is acceptable; reserve the resolvent routing (§2) for the *inverse*-space difference only.
- Underflow masking generalizes to dropping null directions of $\Phi$ (zero/near-zero eigenvalues) rather than individual bins, if the model produces a rank-deficient block.

Note the conditioning caveat for dense $\Phi$: forming $C=\Phi^{-1}+G$ now needs a $\Phi$ factorization, and for HD the cross-pulsar coupling makes $C$ dense across pulsars — i.e. the per-call factorization is the full $(n_p\!\cdot\!m)\times(n_p\!\cdot\!m)$ object. That is an orthogonal cost concern from the precision concern this document addresses; the reference+delta scheme reduces it to one f64 reference factorization plus per-call f32 factorizations, but does not by itself exploit the HD block structure. Use the $w=(I+G\Phi)^{-1}v$ form (§2) if you prefer to avoid materializing $\Phi^{-1}$.

---

## 6. Keeping $C(\theta)$ factorizable in f32 — the $G$ reconstruction (Sec. V)

The f32 Cholesky in §2 needs $C(\theta)=D+G$ to stay positive definite. In f32, $G=F^{\mathsf T}N^{-1}F$ can pick up tiny **negative** eigenvalues (it is PSD by construction, but rounding and a near-singular regime where the GP signal $\to0$ break this). A negative eigenvalue makes the Cholesky fail or amplifies error by the condition number.

Fix: replace $G$ by its nearest PSD matrix in Frobenius norm — eigendecompose, clip, reconstruct:

$$
G = Z\Lambda Z^{\mathsf T},\qquad \Lambda' = \max(\Lambda,0),\qquad G' = Z\Lambda' Z^{\mathsf T}.
$$

$G'$ is the Frobenius-closest PSD matrix to $G$. Under **fixed WN**, $G$ is $\theta$-independent: do this clip **once in f64** and fold $G'$ in as the constant used everywhere $G$ appears (§2, §3). Under sampled WN it must be redone per WN update (see §7). This is what lets the per-call $C(\theta)$ factorization stay in f32 without NaNs.

---

## 7. Sampled white noise (the same expansion, with a different ceiling)

Everything above froze $N$. When $N$ is sampled, $v$, $G$, `ytNmy`, $\ln\det N$ all become per-call and the change is spread over all $n_{\rm toa}$ points, so the change is no longer confined to the small $n_{\rm gp}$ space. But it is **the same family of expansion** — a reference + delta with a per-element merge and a resolvent — and it is exact. The one real difference from the Φ case is the *ceiling* on the delta (below).

**Reference-whitened coordinates.** Whiten by the f64 reference white noise once: $\tilde y = N_{\rm ref}^{-1/2}y$, $\tilde F = N_{\rm ref}^{-1/2}F$ (so $\tilde F^{\mathsf T}\tilde F = G_0$). $N$ is diagonal (no ECORR; with ECORR everything below goes per-epoch-block instead of per-element), and the WN model is a low-dimensional parametrization, $N_i=\mathrm{EFAC}^2(\sigma_i^2+\mathrm{EQUAD}^2)$. Define the WN ratio and its analytic increment

$$
r_i \equiv \frac{N_{{\rm ref},i}}{N_i},\qquad r_i-1 = -\frac{\Delta N_i}{N_i},\quad \Delta N_i = N_i-N_{{\rm ref},i}\ \text{(closed form, e.g. }\mathrm{EFAC}^2(\mathrm{EQUAD}^2-\mathrm{EQUAD}_{\rm ref}^2)).
$$

Forming $r_i-1$ this way avoids the inner big-minus-big ($N_i^{-1}\sim10^{14}$); the reference is $r_i\equiv1$.

**(A) WN-only part — merge quadratic and logdet per TOA.** The would-be likelihood with no GP is $\ln L_{\rm WN}=-\tfrac12[y^{\mathsf T}N^{-1}y+\ln\det N+n\ln2\pi]$. Its increment, combined *per element before summing*, is

$$
\boxed{\;\Delta\ln L_{\rm WN} = -\tfrac12\sum_i\big[(r_i-1)\,\tilde y_i^2 - \ln r_i\big]\;}
$$

This is the direct analog of the logdetΦ+logdetC merge, and the merge is what makes it f32-safe. Taken separately, $\sum_i(r_i-1)\tilde y_i^2$ and $\sum_i\ln r_i$ each have mean $\sim n\bar\epsilon$ ($\approx10^4$ for a 1% WN move), of opposite sign — reduce-then-subtract reintroduces the cancellation. Merged, the first-order means cancel and what is left is a KL divergence between the two WN models:

$$
\mathbb E[\Delta\ln L_{\rm WN}] \approx -\tfrac14\sum_i(r_i-1)^2 \sim n\epsilon^2,
\qquad
\operatorname{sd} \approx \sqrt{\tfrac12\textstyle\sum_i(r_i-1)^2}\sim\sqrt n\,\epsilon .
$$

For $n=10^6$: mean $\approx1$ at $\epsilon\sim10^{-3}$, $\approx10^3$ at $\epsilon\sim$ few %; sd $\approx1$–$30$ over the same range. So the reduction targets $10^0$–$10^3$, **not** $10^6$, and the summands are themselves $\approx\tfrac12\epsilon(\tilde y_i^2-1)$ — mean-zero, $\mathcal O(\epsilon)$ — so partial sums never climb to $10^6$ and the accumulator is well-conditioned. A tree/pairwise reduction (XLA default) resolves it in f32; f64-accumulate is cheap insurance, not a requirement.

**(B) GP-explained part — $v$ and $G$ move with $N$, via the same resolvent.** The GP correction is $\ln L_{\rm GP}=\tfrac12 v^{\mathsf T}C^{-1}v-\tfrac12\ln\det(I+\Phi G)$, with $v(N)=\tilde F^{\mathsf T}W\tilde y$, $G(N)=\tilde F^{\mathsf T}W\tilde F$, $W=\operatorname{diag}(r_i)$. Their increments carry the small weight $W-I$ inside, so they are directly formed (no big-minus-big) and small:

$$
\Delta v = \tilde F^{\mathsf T}(W-I)\tilde y,\qquad \Delta G = \tilde F^{\mathsf T}(W-I)\tilde F .
$$

These are data-sized contractions (length-$n$ reductions, f64-accumulated, done once per WN update), but with weight $r_i-1\sim\epsilon$ the results are small. Now $C=\Phi^{-1}+G=C_0+\Delta K$ carries **two** perturbations,

$$
\Delta K = \underbrace{(\Phi^{-1}-\Phi_{\rm ref}^{-1})}_{\Delta D,\ \text{from }\Phi} + \underbrace{(G-G_0)}_{\Delta G,\ \text{from }N},
$$

and §2–§3 generalize by replacing the single perturbation with $\Delta K$ and accounting for $\Delta v$:

$$
\Delta\mathcal Q_{\rm GP} = \Delta v^{\mathsf T}\mu + v_0^{\mathsf T}\big[C^{-1}(\Delta v - \Delta K\,\mu_0)\big],\qquad \mu=C^{-1}v,\ \mu_0=C_0^{-1}v_0,
$$
$$
\Delta\mathcal L_{\det} = \ln\det\!\Big(I + S_0^{-1}\big(\Phi_{\rm ref}\,\Delta G + \Delta\Phi\,G\big)\Big),\qquad S_0=\Phi_{\rm ref}C_0 .
$$

Both reduce to the fixed-WN formulas (§2 boxed $\Delta\mathcal Q$, §3 boxed $\Delta\mathcal L_{\det}$) when $\Delta G=0$, $\Delta v=0$. Everything past the $\Delta v$/$\Delta G$ contractions is small-$n_{\rm gp}$-space and f32-safe. Then $\Delta\ln L = \Delta\ln L_{\rm WN} + \tfrac12\Delta\mathcal Q_{\rm GP} - \tfrac12\Delta\mathcal L_{\det}$.

**The one genuine asymmetry with Φ.** The GP delta is bounded $\sim\sqrt m$ *regardless* of where Φ sits, because the GP is subdominant ($\|\Phi\|_F\ll\|N\|_F$). The WN delta has no such ceiling — it *is* the KL between the two WN models, growing as $n\epsilon^2$ (mean) / $\sqrt n\,\epsilon$ (sd) with the excursion. Near a good WN reference it is f32-comfortable; for prior-wide WN moves you refresh the reference (recompute the f64 baseline + re-clip $G\to G'$, §6) or run the WN block in f64. Practically WN is a separate, less-frequent Gibbs block, so the data-sized $\Delta v$/$\Delta G$ matmuls are amortized and a periodic reference refresh is natural.

**Two non-negotiables for the WN block:** (i) merge quadratic and logdet *per TOA before reducing* — reduce-then-subtract throws away the cancellation; (ii) form $r_i-1$, $\Delta v$, $\Delta G$ with the weight already inside, never as a difference of two large reductions.

---

## 8. Implementation mapping (graph / folding / pins)

Express the whole thing as **one symbolic graph**; constant-folding does the fixed-WN-vs-sampled-WN split for you.

- Write $\ln L(\theta_{\rm ref})$ and its sub-pieces ($v^{\mathsf T}u_{\rm ref}$, $\ln\det\Phi_{\rm ref}$, $\ln\det C(\theta_{\rm ref})$, `ytNmy`, $\ln\det N$) **explicitly** in the graph. Under fixed WN they are $\theta$-independent and fold into **f64 constants** at compile time. So do $v$, $G$ (use $G'$, §6) and $L_0$, $w_{\rm ref}$, the $\Phi_{\rm ref}$ factorization.
- New **f64-pinned** nodes: the reference baselines $\ln L(\theta_{\rm ref})$, $v^{\mathsf T}u_{\rm ref}$, $L_0$, $w_{\rm ref}$, the $\Phi_{\rm ref}$/$S_0$ factor. Keep the existing f64 pins on `ytNmy`, $\ln\det N$.
- **f32** per call: the per-call factorization (Cholesky of $C(\theta)$, or LU of $I+G\Phi$), the solve for $w$, the §2 contraction $w^{\mathsf T}\Delta\Phi\,w_{\rm ref}$, the §3 `slogdet(I+M)`. Build $\Delta\Phi$ from the model (log-space for diagonal PSD, §5; block-scaled $\Gamma$ for HD; direct for fftcov).
- Final assembly: $\ln L(\theta) = \ln L(\theta_{\rm ref})_{\,\rm f64} + \big(\tfrac12\Delta\mathcal Q - \tfrac12\Delta\mathcal L_{\det}\big)_{\,\rm f32}$, with the final scalar add done in f64 (cheap).

**Deliverable summary for Code.** Implement two per-call f32 increments —
$\Delta\mathcal Q = w^{\mathsf T}\Delta\Phi\,w_{\rm ref}$ (§2, general $\Phi$, no diagonal assumption) and
$\Delta\mathcal L_{\det}=\operatorname{slogdet}(I+S_0^{-1}\Delta\Phi G)$ (§3, **merged** items 2+3, reusing $L_0$) — on top of an f64 baseline $\ln L(\theta_{\rm ref})$; precompute the §4 table once in f64; handle the GP prior as a general SPD $\Phi$ that may be diagonal (per-bin PSD; §5 log-space + underflow masking), dense within a pulsar (fftcov), or dense across pulsars (HD); clip $G\to G'$ for factorizability (§6); and for sampled WN use the same expansion (§7) — per-TOA merge of the WN quadratic and logdet (whose delta is a KL, $\sim n\epsilon^2$ / $\sqrt n\epsilon$, not $10^6$), plus $(W-I)$-weighted $\Delta v,\Delta G$ feeding the §2–§3 increments through the combined perturbation $\Delta K=\Delta D+\Delta G$, with a reference refresh when WN roams far.