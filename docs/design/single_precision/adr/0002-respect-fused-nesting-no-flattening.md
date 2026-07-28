# Reference+delta respects the fused per-pulsar nesting (no flattening)

Reference+delta is applied to the nested (fused) Woodbury, not to an equivalent
flattened single Woodbury over a block-`Φ` of all sampled GPs. The two are
mathematically identical, but flattening replaces `Np` batched per-pulsar `m×m`
factorizations with one dense `(Np·m)²` factorization — destroying the batched
intrinsic-red-noise marginalization that is the entire GPU win at array scale
(`Np ~ 67`). So flattening is rejected as a non-starter, not a tradeable optimization.

Consequence: the increment must be derived for the nested case, where the outer
(cross-pulsar) level sees a *sampled* effective noise (the inner per-pulsar red-noise
GP is sampled). This nested-projection increment is on the critical path. It is
bounded, not open-ended: it is the §2 inner resolvent increment (batched per pulsar),
propagated through the linear projection, fed as the `Δv`/`ΔG` of the §7 two-perturbation
increment at the outer level — the same structure as the §7 sampled-white-noise result,
with the inner GP as the perturbation source instead of white noise.
