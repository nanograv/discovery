"""Single-pulsar outlier detection (Wang & Taylor 2022) on top of
`discovery.PulsarLikelihood`.

See `nanograv_single_pulsar_outlier.md` for design notes and the
statistical model.
"""

import dataclasses
import functools
import re

import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
import numpyro.infer

from .. import likelihood
from .. import prior
from .. import signals


# Standard discovery priors + outlier-specific knobs.
# `nu` is the inverse-gamma dof for the per-TOA alpha_i scaling.
# `theta_m` is the Tak/Ellis/Ghosh prior mean for the outlier fraction
# (a scalar hyperparameter, not a sampled range).
priordict_outlier_default = {
    **prior.priordict_standard,
    "nu":      [1, 40],
    "theta_m": 0.01,
}


def make_outlier_likelihood(psr, *,
                            Tspan=None,
                            psd=None,
                            components: int = 30,
                            rn_name: str = "red_noise",
                            noisedict=None):
    """Build a `PulsarLikelihood` configured for the outlier analysis.

    `outliers=True` is set on the measurement noise (introduces the per-TOA
    `alpha_scaling` parameter), and `variable=True` on the timing and ECORR
    GPs (required so their coefficients are not analytically marginalized
    away — the outlier model needs to draw them via `sample_conditional`).

    Args:
        psr: discovery `Pulsar` object.
        Tspan: float seconds for the Fourier basis. Defaults to
            `signals.getspan(psr)`. Pass the full PTA span when these
            results will be combined with a PTA-wide analysis.
        psd: red-noise PSD function. Defaults to `signals.freespectrum`.
        components: int, number of Fourier components for the red-noise GP.
        rn_name: name of the red-noise GP (controls parameter naming).
        noisedict: optional dict of fixed white-noise values. Defaults to
            empty — the outlier analysis treats WN as variable.

    Returns:
        A configured `PulsarLikelihood`.
    """
    if Tspan is None:
        Tspan = signals.getspan(psr)
    if psd is None:
        psd = signals.freespectrum
    if noisedict is None:
        noisedict = {}

    return likelihood.PulsarLikelihood([
        psr.residuals,
        signals.makenoise_measurement(psr, noisedict=noisedict, outliers=True),
        signals.makegp_timing(psr, svd=True, variable=True),
        signals.makegp_ecorr(psr, variable=True),
        signals.makegp_fourier(psr, psd, components, T=Tspan, name=rn_name),
    ])


def _partition_params(psrl):
    """Bucket `psrl.logL.params` into outlier-model categories.

    Args:
        psrl: `PulsarLikelihood` built for the outlier analysis (must
            have been constructed with `outliers=True`).

    Returns:
        dict with sorted lists under keys `efac`, `equad`, `ecorr`,
        `red_noise`, plus a single string under `alpha_scaling`.

        The ``red_noise`` bucket contains *all* GP hyperparameters that
        are not white-noise or ``alpha_scaling`` — i.e. red noise, DM GP,
        solar-wind GP, chromatic GP, etc. The key name is kept as
        ``"red_noise"`` for backward compatibility; downstream code that
        iterates over it by parameter name is unaffected.

    Raises:
        ValueError: if there isn't exactly one `alpha_scaling` parameter
            (usually means `outliers=True` was not passed to
            `makenoise_measurement`).
    """
    params = psrl.logL.params

    efac  = sorted(p for p in params if "efac"          in p)
    equad = sorted(p for p in params if "equad"         in p)
    ecorr = sorted(p for p in params if "log10_ecorr"   in p)
    alpha = [p for p in params if "alpha_scaling" in p]

    if len(alpha) != 1:
        raise ValueError(
            f"expected exactly one alpha_scaling param, found {alpha}; "
            "did you pass outliers=True to makenoise_measurement?")

    # Everything that is not a WN parameter or alpha_scaling is a GP
    # hyperparameter (red noise, DM GP, SW GP, chromatic GP, …).  Using
    # set-difference keeps the classifier future-proof without any
    # hard-coded name patterns.
    _classified = set(efac) | set(equad) | set(ecorr) | {alpha[0]}
    gp_hypers = sorted(p for p in params if p not in _classified)

    return {
        "efac":         efac,
        "equad":        equad,
        "ecorr":        ecorr,
        "red_noise":    gp_hypers,   # key kept for backward compatibility
        "alpha_scaling": alpha[0],
    }


def assemble_pardict(hmc_sites: dict, partition: dict) -> dict:
    """Expand the array-valued numpyro sites into the per-name scalar
    dict that `psrl` expects.

    Args:
        hmc_sites: current HMC state. Must contain array sites `efacs`,
            `equads`, `ecorrs` (each one element per partition entry),
            and any already-named scalar/vector sites (e.g. `nu`,
            red-noise hypers keyed by their full name).
        partition: output of `_partition_params(psrl)`.

    Returns:
        dict keyed by full parameter names, suitable for `psrl.logL(...)`.
    """
    pardict = dict(hmc_sites)
    pardict.update(zip(partition["efac"],  hmc_sites["efacs"]))
    pardict.update(zip(partition["equad"], hmc_sites["equads"]))
    pardict.update(zip(partition["ecorr"], hmc_sites["ecorrs"]))
    return pardict


# ---- Gibbs draws ----

def draw_theta(rng_key, z_i, k: float,
               m: float = 0.01, eps: float = 1e-6):
    """Beta-conjugate Gibbs update for the outlier-fraction mixture weight.

    Prior: `Beta(k*m, k*(1-m))`, where `k` is the prior pseudo-count and
    `m` the prior mean. With `n = sum(z_i)` Bernoulli successes out of
    `N = z_i.size` trials the posterior is
    `Beta(k*m + n, k*(1-m) + N - n)`. The draw is clipped away from 0
    and 1 so downstream log-space computations stay finite. The
    Tak/Ellis/Ghosh convention is to set `k = N`; the caller picks.

    Args:
        rng_key: PRNG key.
        z_i: int array of current outlier indicators, shape (N,).
        k: prior pseudo-count (Beta-prior strength hyperparameter).
        m: prior mean for theta. Default 0.01.
        eps: clip distance away from {0, 1}.

    Returns:
        Scalar theta in (eps, 1-eps).
    """
    N = z_i.shape[0]
    z_sum = jnp.sum(z_i)
    alpha = k * m + z_sum
    beta  = k * (1.0 - m) + N - z_sum
    theta = jax.random.beta(rng_key, alpha, beta)
    return jnp.clip(theta, eps, 1.0 - eps)


def draw_alpha(rng_key, z_i, nu: float, yprime, sigma2_unit):
    """Per-TOA InverseGamma Gibbs update for the outlier scale `alpha_i`.

    For each TOA i the conditional is
    `alpha_i ~ InverseGamma((nu + z_i) / 2, (nu + z_i * (yp_i^2 / s_i^2)) / 2)`,
    sampled as `rate / Gamma(shape)`. When `z_i = 0` the update collapses
    to the InverseGamma(nu/2, nu/2) prior, independent of the residual.

    Args:
        rng_key: PRNG key.
        z_i: int array of current outlier indicators, shape (N,).
        nu: scalar dof of the alpha prior.
        yprime: residual after subtracting the GP mean (y - T @ coeffs),
            shape (N,).
        sigma2_unit: per-TOA unscaled variance `sigma_i^2(theta_w)`, shape (N,).

    Returns:
        alpha_i: float array, shape (N,), all positive.
    """
    per_toa_quad = (yprime ** 2) / sigma2_unit
    shape = 0.5 * (nu + z_i)
    rate  = 0.5 * (nu + z_i * per_toa_quad)
    return rate / jax.random.gamma(rng_key, shape)


def draw_z(rng_key, theta: float, yprime, sigma2_scaled, sigma2_unit):
    """Per-TOA Bernoulli Gibbs update for the outlier indicators `z_i`.

    Computes the mixture posterior probability in log-space:

        p(z_i=1) ∝ theta     * N(yp_i | 0, alpha_i * sigma_i^2)
        p(z_i=0) ∝ (1-theta) * N(yp_i | 0,            sigma_i^2)

    then draws `z_i ~ Bernoulli(q_i)` where `q_i = p(z_i=1 | ...)`.

    Args:
        rng_key: PRNG key.
        theta: scalar mixture weight (caller should pass it already
            clipped away from 0/1).
        yprime: residual after subtracting the GP mean (y - T @ coeffs),
            shape (N,).
        sigma2_scaled: per-TOA variance with the outlier scaling baked
            in (`alpha_scaling = alpha_i`), shape (N,).
        sigma2_unit: per-TOA variance without scaling
            (`alpha_scaling = 1`), shape (N,).

    Returns:
        z_i: int32 array of {0, 1}, shape (N,).
        q:   float array of mixture probabilities, shape (N,).
    """
    yp_sq = yprime ** 2
    # Gaussian log-pdf up to the constant log(2pi)/2 -- cancels in log_num - log_alt.
    logp_scaled = -0.5 * (jnp.log(sigma2_scaled) + yp_sq / sigma2_scaled)
    logp_unit   = -0.5 * (jnp.log(sigma2_unit)   + yp_sq / sigma2_unit)

    log_num = jnp.log(theta)    + logp_scaled
    log_alt = jnp.log1p(-theta) + logp_unit
    q = jnp.exp(log_num - jnp.logaddexp(log_num, log_alt))
    q = jnp.clip(q, 0.0, 1.0)

    z_i = jax.random.bernoulli(rng_key, q).astype(jnp.int32)
    return z_i, q


def draw_coeffs(rng_key, pardict: dict, sample_cond_fn, cvars):
    """Draw GP coefficients from the analytic conditional `p(b | ..., dt)`.

    Thin wrapper over `psrl.sample_conditional` that also returns the
    coefficients flattened in `cvars` order (for the downstream
    `T @ coeffs` matvec). The caller is responsible for having set
    `pardict[alpha_key]` to `alpha_i ** z_i` so the current mixture
    state is baked into the noise.

    Args:
        rng_key: PRNG key.
        pardict: full parameter dict for `psrl.logL` (WN + RN hypers +
            alpha_scaling already set per the mixture state).
        sample_cond_fn: a callable `(key, params) -> (key, coeffs_dict)`,
            typically `psrl.sample_conditional`.
        cvars: list of coefficient parameter names defining the order
            used to flatten (the order of `psrl.N.index`).

    Returns:
        coeffs_dict: dict of per-component coefficient arrays.
        coeffs_flat: 1-D array, `jnp.hstack([coeffs_dict[c] for c in cvars])`.
    """
    _, coeffs_dict = sample_cond_fn(rng_key, pardict)
    coeffs_flat = jnp.hstack([coeffs_dict[c] for c in cvars])
    return coeffs_dict, coeffs_flat


# ---- numpyro model ----

def _lookup_prior(name, priordict):
    """Look up a (low, high) range in priordict by exact key or regex match."""
    for pat, rng in priordict.items():
        if pat == name or re.match(pat, name):
            return rng
    raise KeyError(f"no prior pattern matched parameter {name!r}")


def make_outlier_model(psrl, *, priordict=None):
    """Build the numpyro model for the single-pulsar outlier analysis.

    HMC samples white-noise hypers (`efacs`, `equads`, `ecorrs`), red-noise
    hypers (whatever sites `psrl.logL.params` exposes for the chosen PSD),
    and the InverseGamma dof `nu`. The Gibbs sites (`coeffs`, `theta`,
    `z_i`, `q`, `alpha_i`) are declared with placeholder distributions so
    they are registered in the trace; `make_outlier_gibbs_fn` overrides
    them at each step. The model also records the marginalized
    `loglike` and the assembled `params` dict as deterministics.

    Args:
        psrl: outlier-configured `PulsarLikelihood`.
        priordict: prior ranges (regex-keyed). Defaults to
            `priordict_outlier_default`. Must contain a match for every
            HMC site, including `nu`.

    Returns:
        A numpyro model callable `model(rng_key=None)`.
    """
    if priordict is None:
        priordict = priordict_outlier_default

    partition  = _partition_params(psrl)
    alpha_key  = partition["alpha_scaling"]
    N          = psrl.y.size
    n_coeffs   = psrl.N.F.shape[-1]

    # Resolve the WN ranges from the first param in each group (all share a
    # pattern in practice, e.g. `(.*_)?efac`).
    efac_range  = _lookup_prior(partition["efac"][0],  priordict)
    equad_range = _lookup_prior(partition["equad"][0], priordict)
    ecorr_range = _lookup_prior(partition["ecorr"][0], priordict)
    nu_range    = priordict["nu"]

    # Red noise: each named site may be scalar (power-law `log10_A`, `gamma`)
    # or vector (free-spectrum `log10_rho(K)`). Detect size from a trailing
    # "(K)" in the name.
    rn_entries = []
    for rn_name in partition["red_noise"]:
        m = re.search(r"\((\d+)\)$", rn_name)
        size = int(m.group(1)) if m else 1
        rn_entries.append((rn_name, size, _lookup_prior(rn_name, priordict)))

    coeff_slices = psrl.N.index

    def model(rng_key=None):
        # HMC sites: WN hypers
        efacs  = numpyro.sample("efacs",
                                dist.Uniform(*efac_range).expand([len(partition["efac"])]))
        equads = numpyro.sample("equads",
                                dist.Uniform(*equad_range).expand([len(partition["equad"])]))
        ecorrs = numpyro.sample("ecorrs",
                                dist.Uniform(*ecorr_range).expand([len(partition["ecorr"])]))
        numpyro.sample("nu", dist.Uniform(*nu_range))

        # HMC sites: red noise (one numpyro site per `psrl` red-noise param)
        rn_samples = {}
        for rn_name, size, rng in rn_entries:
            d = dist.Uniform(*rng)
            rn_samples[rn_name] = numpyro.sample(rn_name,
                                                 d.expand([size]) if size > 1 else d)

        # Gibbs placeholders. Priors don't enter the HMC conditional given
        # the gibbs sites, so any registered distribution works here.
        coeffs  = numpyro.sample("coeffs",
                                 dist.Uniform(-1e-4, 1e-4).expand([n_coeffs]))
        numpyro.sample("theta", dist.Uniform(0, 1))
        z_i     = numpyro.sample("z_i", dist.Binomial(1, 0.5).expand([N]))
        numpyro.sample("q",     dist.Uniform(0, 1).expand([N]))
        alpha_i = numpyro.sample("alpha_i", dist.Uniform(0, 100).expand([N]))

        # Assemble the full pardict for psrl.logL.
        pardict = {}
        pardict.update(zip(partition["efac"],  efacs))
        pardict.update(zip(partition["equad"], equads))
        pardict.update(zip(partition["ecorr"], ecorrs))
        pardict.update(rn_samples)
        pardict[alpha_key] = alpha_i ** z_i
        pardict.update({k: coeffs[slc] for k, slc in coeff_slices.items()})

        logl = numpyro.deterministic("loglike", psrl.logL(pardict))
        numpyro.deterministic("params", pardict)
        numpyro.factor("logl", logl)

    return model


# ---- Orchestrator ----

def make_outlier_gibbs_fn(psrl):
    """Build the Gibbs kernel for the single-pulsar outlier analysis.

    Reads `psrl`-specific static bits once and returns a closure that
    performs one round of Gibbs updates in the order
    coeffs -> theta -> z_i -> alpha_i. All mixture-state bookkeeping
    (which value to put in `pardict[alpha_key]` at each step) lives in
    the closure; each `draw_*` it calls is a pure function.

    Args:
        psrl: outlier-configured `PulsarLikelihood` (built via
            `make_outlier_likelihood` or an equivalent setup with
            `outliers=True` and `variable=True` on the GPs).

    Returns:
        gibbs_fn: `(rng_key, gibbs_sites, hmc_sites) -> gibbs_sites_new`,
            matching numpyro's `HMCGibbs.gibbs_fn` contract.
    """
    partition      = _partition_params(psrl)
    alpha_key      = partition["alpha_scaling"]
    sample_cond_fn = psrl.sample_conditional
    cvars          = list(psrl.N.index.keys())

    T = psrl.N.F
    y = psrl.y
    N = y.size
    ones_N = jnp.ones(N)

    make_Nalpha = psrl.N.N_var.getN

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        k_c, k_t, k_z, k_a = jax.random.split(rng_key, 4)
        pardict = assemble_pardict(hmc_sites, partition)

        # 1. coeffs: bake current mixture state alpha^z into the diagonal
        pardict[alpha_key] = gibbs_sites["alpha_i"] ** gibbs_sites["z_i"]
        _, coeffs_flat = draw_coeffs(k_c, pardict, sample_cond_fn, cvars)

        means  = T @ coeffs_flat
        yprime = y - means

        # 2. theta (Tak/Ellis/Ghosh prior strength k = N)
        theta = draw_theta(k_t, gibbs_sites["z_i"], N)

        # 3. z_i: needs both alpha-scaled and unit-scaled diagonals
        pardict[alpha_key] = gibbs_sites["alpha_i"]
        sigma2_scaled = make_Nalpha(pardict)
        pardict[alpha_key] = ones_N
        sigma2_unit   = make_Nalpha(pardict)
        z_i, q = draw_z(k_z, theta, yprime, sigma2_scaled, sigma2_unit)

        # 4. alpha_i (with the freshly-drawn z_i and the unit-variance diagonal)
        alpha_i = draw_alpha(k_a, z_i, hmc_sites["nu"], yprime, sigma2_unit)

        return {
            "coeffs":  coeffs_flat,
            "theta":   theta,
            "z_i":     z_i,
            "alpha_i": alpha_i,
            "q":       q,
        }

    return gibbs_fn


# ---- Runner + result wrapper ----

def _init_values_from_priordict(psrl, priordict):
    """Pick HMC-site init values at the midpoint of each prior range."""
    partition = _partition_params(psrl)

    def mid(name):
        a, b = _lookup_prior(name, priordict)
        return 0.5 * (a + b)

    init = {
        "efacs":  jnp.full(len(partition["efac"]),  mid(partition["efac"][0])),
        "equads": jnp.full(len(partition["equad"]), mid(partition["equad"][0])),
        "ecorrs": jnp.full(len(partition["ecorr"]), mid(partition["ecorr"][0])),
        "nu":     0.5 * (priordict["nu"][0] + priordict["nu"][1]),
    }
    for rn_name in partition["red_noise"]:
        m = re.search(r"\((\d+)\)$", rn_name)
        size = int(m.group(1)) if m else 1
        midval = mid(rn_name)
        init[rn_name] = jnp.full(size, midval) if size > 1 else midval
    return init


@dataclasses.dataclass
class OutlierFitResult:
    """Wrapper around the fitted MCMC + the inputs needed for post-processing.

    Attributes:
        mcmc: the fitted `numpyro.infer.MCMC` object.
        psrl: the `PulsarLikelihood` used for the fit.
        psr:  the underlying `discovery.Pulsar`.
    """
    mcmc: object
    psrl: object
    psr:  object

    @functools.cached_property
    def samples(self):
        """All posterior samples, as returned by `mcmc.get_samples()`."""
        return self.mcmc.get_samples()

    def outlier_mask(self, threshold: float = 0.1):
        """Per-TOA outlier mask: `<z_i> > threshold` across samples.

        Args:
            threshold: classification cutoff for the per-TOA average of
                `z_i`. Default 0.1 (Wang & Taylor 2022).

        Returns:
            Boolean numpy array, shape `(n_toa,)`.
        """
        return np.mean(np.asarray(self.samples["z_i"]), axis=0) > threshold

    def whitened_residuals(self, sample_idx: int):
        """Whitened residuals `(y - T @ b_hat) / sigma_unit` at one sample.

        Args:
            sample_idx: index into the posterior samples.

        Returns:
            numpy array of shape `(n_toa,)`. Uses the conditional-mean
            coefficients and the unit-alpha noise diagonal (so the user
            can directly compare with `N(0, 1)`).
        """
        params = {k: np.asarray(v[sample_idx])
                  for k, v in self.samples["params"].items()}
        coeffs, _ = self.psrl.conditional(params)
        mean = self.psrl.N.F @ coeffs
        yp = self.psr.residuals - mean

        partition = _partition_params(self.psrl)
        params[partition["alpha_scaling"]] = np.ones(self.psr.residuals.size)
        sigma2 = self.psrl.N.N_var.getN(params)
        return np.asarray(yp / jnp.sqrt(sigma2))

    def summary(self):
        """`numpyro.diagnostics.summary` over the gathered samples.

        Drops the `params` deterministic (a nested dict) which numpyro's
        summary cannot consume.
        """
        flat = {k: v for k, v in self.samples.items() if not isinstance(v, dict)}
        return numpyro.diagnostics.summary(flat, group_by_chain=False)

    def rhat(self, param=None):
        """Rhat for a single site name, or a dict over all sites."""
        s = self.summary()
        if param is None:
            return {k: v["r_hat"] for k, v in s.items()}
        return s[param]["r_hat"]

    def traceplot(self, param: str):
        """Quick trace plot for one site. Requires matplotlib.

        Args:
            param: site name to plot.

        Returns:
            `(fig, ax)` matplotlib pair.
        """
        import matplotlib.pyplot as plt
        values = np.asarray(self.samples[param])
        fig, ax = plt.subplots(figsize=(8, 3))
        if values.ndim == 1:
            ax.plot(values)
        else:
            for i in range(min(5, values.shape[-1])):
                ax.plot(values[..., i], label=f"{param}[{i}]")
            ax.legend()
        ax.set_xlabel("iteration")
        ax.set_ylabel(param)
        return fig, ax


def run_outlier_mcmc(psr, *,
                     rng_key,
                     psrl=None,
                     priordict=None,
                     init_values=None,
                     num_warmup: int = 100,
                     num_samples: int = 100,
                     max_tree_depth: int = 6,
                     target_accept_prob: float = 0.8,
                     Tspan=None,
                     components: int = 30):
    """Run HMCGibbs end-to-end for the single-pulsar outlier analysis.

    Builds `psrl` if not supplied, builds the model + Gibbs kernel,
    initializes HMC sites at prior midpoints (overridden by `init_values`),
    and runs `numpyro.infer.MCMC`. Returns an `OutlierFitResult`.

    Args:
        psr: `discovery.Pulsar`.
        rng_key: PRNG key (e.g. `jax.random.key(seed)`).
        psrl: optional pre-built outlier `PulsarLikelihood`. If `None`,
            `make_outlier_likelihood(psr, Tspan=Tspan, components=components)`
            is used.
        priordict: prior dict. Defaults to `priordict_outlier_default`.
        init_values: optional dict of HMC-site init values; missing keys
            are filled from prior midpoints.
        num_warmup, num_samples: MCMC iteration counts.
        max_tree_depth, target_accept_prob: NUTS knobs.
        Tspan, components: forwarded to `make_outlier_likelihood` only
            when `psrl` is `None`.

    Returns:
        `OutlierFitResult`.
    """
    if priordict is None:
        priordict = priordict_outlier_default
    if psrl is None:
        psrl = make_outlier_likelihood(psr, Tspan=Tspan, components=components)

    model    = make_outlier_model(psrl, priordict=priordict)
    gibbs_fn = make_outlier_gibbs_fn(psrl)

    init = _init_values_from_priordict(psrl, priordict)
    if init_values:
        init.update(init_values)
    init_strategy = numpyro.infer.util.init_to_value(values=init)

    nuts = numpyro.infer.NUTS(model,
                              init_strategy=init_strategy,
                              max_tree_depth=max_tree_depth,
                              target_accept_prob=target_accept_prob)
    kernel = numpyro.infer.HMCGibbs(
        nuts,
        gibbs_fn=jax.jit(gibbs_fn),
        gibbs_sites=["theta", "z_i", "alpha_i", "coeffs", "q"],
    )
    mcmc = numpyro.infer.MCMC(kernel,
                              num_warmup=num_warmup,
                              num_samples=num_samples)
    mcmc.run(rng_key)

    return OutlierFitResult(mcmc=mcmc, psrl=psrl, psr=psr)
