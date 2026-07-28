# The reference is a frozen covariance set, not a parameter point

The reference+delta likelihood is expanded around a set of frozen reference
covariances `{N_ref, Φ_ref,k}` — fed to the metamath kernel as constant leaves —
not around a parameter point `θ_ref`. We chose this because the increment math only
ever consumes the covariances (e.g. `L₀ = chol(Φ_ref⁻¹ + G)`, `ΔΦ = Φ(θ) − Φ_ref`),
never the parameters that produced them, and because a good reference `Φ` can come
from a per-pulsar Gibbs / single-pulsar bootstrap that corresponds to no single `θ`
under the model's PSD. A `θ_ref → covariance` evaluation is therefore a thin
convenience layer at the likelihood boundary, not the primitive.

Consequences: a reference covariance is supplied only for the *sampled* GPs (fixed
GPs fold to constants and need none); the same mechanism extends to `N_ref` for the
sampled-white-noise case; the WN reference may need periodic refresh (its delta is a
KL with no ceiling), while a GP `Φ_ref` does not.

Realization in the fused path: the fused pair takes **two** reference covariance
leaves — `Φ_ref,irn` (block-diagonal per pulsar) at `vectorwoodburyjointsolve`, and
`Φ_ref,gw` at `globalwoodbury_fused`; they compose through the projection (ADR 0002).
A `θ_ref → {Φ_ref,…}` evaluation and a hand-supply path both live in the top layer and
never reach metamath — that is what keeps the parameter-sourced option from leaking
parameter handling into the kernel.
