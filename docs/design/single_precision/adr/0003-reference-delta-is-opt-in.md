# Reference+delta is opt-in; absent a reference the likelihood is unchanged

Reference+delta (Piece 2 "Half B") is **opt-in**. If no reference covariance is supplied,
the likelihood graph is exactly today's path (the Half-A f64 final combination) —
byte-identical, no behavior change. The kernel selects the current `woodbury` family when
no reference is given, and the `woodbury_refdelta` family only when a reference is present
(`Pinv_ref is None` → current behavior).

We chose this because people already run the likelihood in float32 *without* any reference
and get usable posteriors on this dataset — so reference+delta is an accuracy boost for the
regimes where the default is not enough (large arrays, tight Hellings–Downs correlations,
NUTS/HMC, evidence), not a mandatory change. Keeping it strictly additive means existing
float32 runs are untouched and there is always a clean fallback.

Consequence: the reference path must be a strict superset of the default. A
"no-reference graph is identical to today" structural test guards it, and the
`θ_ref → Φ_ref` evaluation stays in a thin top layer so the kernel never sees parameters
(ADR 0001). Whether reference+delta is needed at all — and how accurate it must be — is to
be set empirically by a float32-vs-float64 *posterior* comparison, not by the absolute logL
error alone.
