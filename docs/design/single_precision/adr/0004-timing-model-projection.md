# Timing-model projection = exact flat-prior marginalization (and it keeps the corrections)

The timing model is marginalized with an (effectively) **flat / improper prior** — in the
code as a `ConstantGP` with prior variance `constant = 1.0e40` (`signals.py:165,191`).
Feeding that $10^{40}$ into the Woodbury small matrix $C = \Phi^{-1} + F^{\mathsf T}N^{-1}F$
is the single worst conditioning offender in float32: the timing block of $\Phi^{-1}$ is
$10^{-40}$, and $10^{40}$ itself **overflows float32** (max $\approx 3.4\times10^{38}$).
The fix (Rutger's, months ago — it was *the* main issue; once it worked, the
`regularize_FtNmF` eigenvalue hack was no longer needed) is to take the flat-prior limit
**analytically**: project the data orthogonal to the timing subspace instead of carrying the
huge prior variance through the solve.

This ADR records the math, and corrects an earlier overcautious claim that projection could
only serve the marginal $\log L$. **It is not restricted to the marginal likelihood — it
serves the conditional / `clogL` path too, and it does not throw the timing corrections
away.**

## Model

Per pulsar, residuals $\mathbf y\in\mathbb R^{n}$, white noise $K$:

$$
\mathbf y = M\boldsymbol\varepsilon + F\mathbf a + \mathbf n,\qquad
\mathbf n\sim\mathcal N(\mathbf 0, K),\qquad
\mathbf a\sim\mathcal N(\mathbf 0, \Phi),\qquad
\boldsymbol\varepsilon\sim\text{“flat”}\ (\sigma_\varepsilon^2\to\infty).
$$

- $M\in\mathbb R^{n\times m_{\rm tm}}$: timing-model design matrix, improper/flat prior on
  $\boldsymbol\varepsilon$.
- $F\in\mathbb R^{n\times m_{\rm gp}}$: Fourier (and other proper-GP) basis, prior $\Phi(\theta)$.
- $K = \operatorname{diag}(\mathbf N) + F_{\rm ec}\operatorname{diag}(\mathbf P)F_{\rm ec}^{\mathsf T}$:
  white noise, optionally with ECORR ($F_{\rm ec}$ the $0/1$ epoch indicators, $\mathbf P$
  per-epoch variances). Without ECORR, $K=\operatorname{diag}(\mathbf N)$.

Whitened quantities use the map $W = K^{-1/2}$ (so $WKW^{\mathsf T}=I$):

$$
\mathbf y_w = W\mathbf y,\qquad A = WM,\qquad B = WF.
$$

## 1. Whitening: applying $K^{-1/2}$

For diagonal $K=\operatorname{diag}(\mathbf N)$ this is just $W=\operatorname{diag}(\mathbf N^{-1/2})$,
$\operatorname{logdet}K=\sum_i\log N_i$.

With ECORR, $K = D + F_{\rm ec}\Lambda F_{\rm ec}^{\mathsf T}$ with
$D=\operatorname{diag}(\mathbf N)$, $\Lambda=\operatorname{diag}(\mathbf P)$. The ECORR
indicators are **orthogonal** (each TOA is in one epoch), so $F_{\rm ec}^{\mathsf T}D^{-1}F_{\rm ec}$
is diagonal with entries

$$
u_k = \mathbf P_k\!\!\sum_{i\in\text{epoch }k}\!\frac{1}{N_i},
$$

and $K$ decouples per epoch. Its inverse square root has the closed form

$$
K^{-1/2} = D^{-1/2} + D^{-1}F_{\rm ec}\,\operatorname{diag}(\alpha_k)\,F_{\rm ec}^{\mathsf T}D^{-1/2},
\qquad
\alpha_k = \frac{(1+u_k)^{-1/2}-1}{u_k},
$$

with the safe limit $\alpha_k\to-\tfrac12$ as $u_k\to0$ (used below $u_k<10^{-14}$). Acting on
a vector $\mathbf x$, with $b_k=\mathbf P_k\sum_{i\in k}x_i/N_i$,

$$
(W\mathbf x)_i = \frac{x_i}{\sqrt{N_i}} + \frac{1}{\sqrt{N_i}}\sum_{k\ni i}\alpha_k b_k,
\qquad
\operatorname{logdet}K = \sum_i\log N_i + \sum_k\log(1+u_k).
$$

This is exactly `SM_whiten_1d_fused` / `SM_whiten_2d_fused` (the $\log\!1p$ form keeps
$\operatorname{logdet}K$ accurate). The same $W$ whitens every column of $M$ and $F$; the
$\operatorname{logdet}K$ is computed once and shared.

## 2. Projection is the $\sigma_\varepsilon^2\to\infty$ limit, exactly

Let $P_A = A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$ be the orthogonal projector onto the whitened
timing span and $Q = I - P_A$. With $\mathbf r_\perp = Q\mathbf y_w$ and $B_\perp = QB$,
marginalizing $\boldsymbol\varepsilon$ and taking $\sigma_\varepsilon^2\to\infty$ gives, up to
an additive constant independent of $\theta$ (so it cancels in any $\Delta\log L$ and in any
posterior over $\theta$):

$$
-2\log L =
\underbrace{\mathbf r_\perp^{\mathsf T}\big(I + B_\perp\Phi B_\perp^{\mathsf T}\big)^{-1}\mathbf r_\perp}_{\text{projected quadratic}}
+ \operatorname{logdet}\!\big(I + B_\perp\Phi B_\perp^{\mathsf T}\big)
+ \underbrace{\operatorname{logdet}(A^{\mathsf T}A)}_{\text{timing Jacobian}}
+ \operatorname{logdet}K .
$$

The inner Woodbury reduces the first two terms to the small $m_{\rm gp}$ space through
$C = \Phi^{-1} + B_\perp^{\mathsf T}B_\perp$:

$$
\mathbf r_\perp^{\mathsf T}(I+B_\perp\Phi B_\perp^{\mathsf T})^{-1}\mathbf r_\perp
= \mathbf r_\perp^{\mathsf T}\mathbf r_\perp
- (B_\perp^{\mathsf T}\mathbf r_\perp)^{\mathsf T}C^{-1}(B_\perp^{\mathsf T}\mathbf r_\perp),
\qquad
\operatorname{logdet}(I+B_\perp\Phi B_\perp^{\mathsf T}) = \operatorname{logdet}\Phi + \operatorname{logdet}C .
$$

With $10^{40}$ gone, $C$ no longer contains the timing block — it is well conditioned and
float32-safe.

**Derivation.** With $K_\varepsilon = K + \sigma_\varepsilon^2 MM^{\mathsf T}$, Woodbury gives
$K_\varepsilon^{-1} = K^{-1} - K^{-1}M(\sigma_\varepsilon^{-2}I + M^{\mathsf T}K^{-1}M)^{-1}M^{\mathsf T}K^{-1}$.
As $\sigma_\varepsilon^2\to\infty$,

$$
K_\varepsilon^{-1}\to K^{-1} - K^{-1}M(M^{\mathsf T}K^{-1}M)^{-1}M^{\mathsf T}K^{-1} = W^{\mathsf T}QW,
$$

the projected precision, and
$\operatorname{logdet}K_\varepsilon = \operatorname{logdet}K + \operatorname{logdet}(M^{\mathsf T}K^{-1}M) + m_{\rm tm}\log\sigma_\varepsilon^2$;
the last term is the dropped constant, and $M^{\mathsf T}K^{-1}M = A^{\mathsf T}A$.
Substituting $W^{\mathsf T}QW$ into the quadratic and the Fourier Woodbury yields the boxed
result. $\blacksquare$

**Why whiten first (float32).** The projected Gram $B_\perp^{\mathsf T}B_\perp$ must be formed
as the inner product of the *already projected* design
$B_\perp = B - A(A^{\mathsf T}A)^{-1}A^{\mathsf T}B$, **not** as $F^{\mathsf T}K^{-1}F$ minus a
correction. The low-frequency Fourier modes overlap the timing polynomials strongly, so
$F^{\mathsf T}K^{-1}F$ has large entries that nearly cancel against the correction —
catastrophic in float32. Doing the subtraction at the well-scaled, $O(1)$ whitened-vector
level and *then* summing squares keeps it safe. This is the whole reason for the whitening
step (§1).

## 3. Compatibility with `clogL` (sampled coefficients)

In the decentered path the Fourier coefficients $\mathbf a$ are sampled and only
$\boldsymbol\varepsilon$ is marginalized. Because $Q$ is linear, marginalizing
$\boldsymbol\varepsilon$ out of $\mathbf y - F\mathbf a$ commutes with subtracting $B\mathbf a$:

$$
Q(\mathbf y_w - B\mathbf a) = \mathbf r_\perp - B_\perp\mathbf a,
$$

so

$$
\mathrm{clogL}(\mathbf a) = -\tfrac12\big\|\mathbf r_\perp - B_\perp\mathbf a\big\|^2
- \tfrac12\big[\operatorname{logdet}K + \operatorname{logdet}(A^{\mathsf T}A)\big]
+ \log p(\mathbf a).
$$

Project once (independent of $\mathbf a$); the sampled-coefficient likelihood then runs on
$\mathbf r_\perp, B_\perp$, with the timing $\operatorname{logdet}$ an additive constant in
$\mathbf a$. So projection is **not** restricted to the marginal $\log L$. (This is where the
earlier "marginal $\log L$ only" claim was wrong.)

## 4. The timing-model corrections are recoverable

Under the flat prior, the conditional mean of the timing coefficients given data and
hyperparameters is the GLS estimate

$$
\hat{\boldsymbol\varepsilon}
= (M^{\mathsf T}K^{-1}M)^{-1}M^{\mathsf T}K^{-1}(\mathbf y - F\hat{\mathbf a})
= (A^{\mathsf T}A)^{-1}A^{\mathsf T}(\mathbf y_w - B\hat{\mathbf a}),
$$

i.e. exactly the regression coefficients the projection already computes (`coeffr` for
$\hat{\mathbf a}=0$, corrected by $-(A^{\mathsf T}A)^{-1}A^{\mathsf T}B\,\hat{\mathbf a}$). The
joint timing×Fourier cross-covariance is
$-(A^{\mathsf T}A)^{-1}A^{\mathsf T}B\,\operatorname{cov}(\hat{\mathbf a})$. So projection does
**not** discard the timing corrections; it produces them. To reproduce the full-Woodbury
`cond` output (`metamath.py:80`, the joint coefficient posterior mean), back-substitute
$\hat{\boldsymbol\varepsilon}$ from the projection pieces after the Fourier solve.

The only thing projection removes is the explicit $10^{40}$ block in one joint Gaussian; the
estimates and covariances it stood for are all recoverable, more accurately, from the
projected solve.

## Decision

- Projection is the **default-quality, float32-safe** way to marginalize the timing model,
  switched on per model by an explicit `makegp_timing(..., project=True)` flag (opt-in, so
  existing models stay byte-identical — ADR 0003's no-surprises stance).
- It serves the **marginal $\log L$ and the `clogL`/decentered path**.
- Timing-model corrections are obtained by back-substituting $\hat{\boldsymbol\varepsilon}$
  from the projection regression coefficients — no separate un-projected solve is required in
  principle. (Implementation may keep the un-projected Woodbury available as the conditional
  path until the back-substitution is wired; that is an implementation choice, not a math
  limitation.)

## Status / provenance

Math is standard GLS / flat-prior marginalization; matches the `_projection_products` and
`SM_whiten_*` implementations on `meyers-academic/discovery@whitening_and_safe_solves`
(matrix backend). A LaTeX manuscript Patrick uploaded earlier may state the same result — not
in this repo; reconcile if re-supplied. For larger-than-NG15 datasets the residual sampled-GP
cancellation still needs reference+delta (ADR 0003, `research_note_nested_increment.md`);
projection handles the timing-model (flat-prior) cancellation, which is the NG15-scale fix.
