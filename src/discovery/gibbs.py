import numpy as np
import discovery as ds
from . import matrix
jnp = matrix.jnp
import jax
import scipy.special
import numpyro
from numpyro import distributions as dist, infer
import re


def gammaincinv_jax(a, y):
    # # find x such that gammainc(a, x) = y
    gigrad = jax.grad(jax.scipy.special.gammainc)
    x = a
    learn_rate = jnp.min(jnp.array([1, (a+1)*y]))
    def body_fun(n, x):
        return x - learn_rate * (y - jax.scipy.special.gammainc(a, x)) / gigrad(a, x)
    x = jax.lax.fori_loop(0, 1000, body_fun, a)
    # for ii in range(1000):
        # xn = x - learn_rate * (y - gammainc(a, x)) / gigrad(a, x)
        # x = xn
    return x

vgammaincinv_jax = jax.vmap(gammaincinv_jax, (None, 0))


def makerho(key, betas, rhomin=(1e-9)**2, rhomax=(1e-4)**2):
    """
    Parameters:
    -----------
    key : PRNGKey
        Random number generator key.
    betas : ndarray
        Array of betas in inverse gamma distribution
    rhomin : float
        Minimum value of rho^2 for timing delay
    rhomax : float
        Maximum value of rho^2 for timing delay

    Returns:
    --------
    ndarray
        Array of rho^2 values for timing delay (PSD at each frequency)
    """
    xmin = jnp.exp(-betas / rhomin) / betas
    xmax = jnp.exp(-betas / rhomax) / betas

    x = xmin + (xmax - xmin) * jax.random.uniform(key, betas.shape)
    return 0.5 * jnp.log10(betas / jnp.log(1.0 / betas / x))

def makeinvgamma(key, betas, alpha=1.0, rhomin=(1e-9)**2, rhomax=(1e-4)**2):
    xmin = jax.scipy.special.gammainc(alpha, betas / rhomax)
    xmax = jax.scipy.special.gammainc(alpha, betas / rhomin)

    x = xmin + (xmax - xmin) * jax.random.uniform(key, betas.shape)
    return 0.5 * jnp.log10(betas / vgammaincinv_jax(alpha, x)) # not in jax...

def sample_rhos(key, psrs, params, cs, invhdorf=None):
    ret = params.copy()
    coefficient_params = [k for k in cs.keys() if 'coefficients' in k and 'gw' not in k and 'ecorr' not in k]


    # we need these to check that anything with a rho
    # also has coefficients. `sample_conditional` will only sample
    # free spectra that are in the global likelihood, not the individual
    # pulsar likelihoods. So for the multi-pulsar case we need to check that the user
    # has supplied those properly
    model_names_non_gw = []
    coeff_model_names_non_gw = []
    for psr in psrs:
        model_names_non_gw.extend([k.split(psr.name)[1].split("log10_rho")[0][1:-1] for k in params if psr.name in k and 'gw' not in k and 'rho' in k])
        coeff_model_names_non_gw.extend([k.split(psr.name)[1].split("coefficients")[0][1:-1] for k in cs.keys() if psr.name in k and 'gw' not in k and 'ecorr' not in k])
    # print('model names:', np.unique(model_names_non_gw))
    # print('coeff names:', np.unique(coeff_model_names_non_gw))
    if not set(np.unique(model_names_non_gw)) == set(np.unique(coeff_model_names_non_gw)):
        raise ValueError("For multi-pulsar runs, all models with a log10_rho parameter must also have a coefficients parameter. `sample_conditional` only samples models in the global likelihood, not the individual pulsar likelihoods. Consider moving the red noise or DM models from individual pulsar likelihoods to the global likelihood using `make_fouriergp_allpsr`.")
    # back to our usually scheduled programming.


    gw_coefficient_params = [k for k in cs.keys() if 'coefficients' in k and 'gw' in k]

    n_rns = [int(int(k.split('(')[1].split(')')[0]) / 2) for k in coefficient_params]
    for psr in psrs:
        copars = [k for k in coefficient_params if psr.name in k]
        for cp in copars:
            a = cs[cp]
            n_rn = int(a.size / 2)
            model_name = cp.split(psr.name)[1].split("coefficients")[0][1:-1]
            betas = 0.5 * (a[::2]**2 + a[1::2]**2)
            key, subkey = ds.matrix.jnpsplit(key)
            rhos_rn = makerho(subkey, betas)
            ret[f'{psr.name}_{model_name}_log10_rho({n_rn})'] = rhos_rn

    # gw parameters

    if len(gw_coefficient_params) > 0:
        # invhdorf = jnp.linalg.inv(ds.matrix.jnparray([[ds.hd_orf(p1.pos, p2.pos) for p1 in psrs] for p2 in psrs]))
        betas_gw = []
        n_gw  = int(cs[gw_coefficient_params[0]].size / 2)
        a = jnp.concatenate([cs[par] for par in gw_coefficient_params])
        for ii in range(n_gw):
            acos, asin = a[2*ii::2*n_gw], a[2*ii+1::2*n_gw]
            betas_gw.append(0.5 * (jnp.dot(acos, jnp.dot(invhdorf, acos)) + jnp.dot(asin, jnp.dot(invhdorf, asin))))

        model_name = gw_coefficient_params[0].split(psrs[0].name)[1].split("coefficients")[0][1:-1]
        key, subkey = ds.matrix.jnpsplit(key)
        ret[f'{model_name}_log10_rho({n_gw})'] = makeinvgamma(subkey, jnp.array(betas_gw), alpha=len(psrs)/1.)
    key, subkey = ds.matrix.jnpsplit(key)
    return key, ret

def get_priordict_range_for_parameter(parname, priordict=ds.priordict_standard):
    for par, range in priordict.items():
        if parname == par or re.match(par, parname):
            return range
    else:
        raise KeyError(f'Parameter {parname} not found in priordict')


def setup_single_psr_hmc_gibbs(psrl, psrs, priordict=ds.priordict_standard,  invhdorf=None, nuts_kwargs={}):
    jlogl = jax.jit(psrl.logL)
    sample_cond = psrl.sample_conditional
    jsc = jax.jit(sample_cond)
    parnames = psrl.logL.params
    gibbs_params = [k for k in parnames if 'log10_rho' in k]
    # get non-gibbs parameters
    non_gibbs_params = list(set(parnames) - set(gibbs_params))
    # number of red noise parameters per pulsar
    n_rns = [int(k.split('(')[1].split(')')[0]) for k in gibbs_params]
    # needed for sample_rhos above
    if not isinstance(psrs, list):
        psrs = [psrs]

    def hmc_gibbs_model():
        pars = {}
        for gp, nrn in zip(gibbs_params, n_rns):
            myrange = get_priordict_range_for_parameter(gp, priordict)
            pars[gp] = numpyro.sample(gp, dist.Uniform(myrange[0], myrange[1]).expand([nrn]))
        for ngp in non_gibbs_params:
            myrange = get_priordict_range_for_parameter(ngp, priordict)
            pars[ngp] = numpyro.sample(ngp, dist.Uniform(myrange[0], myrange[1]))
        numpyro.factor('logL', jlogl(pars))

    def gibbs_step(rng_key, gibbs_sites, hmc_sites):
        key, cs = jsc(rng_key, {**gibbs_sites, **hmc_sites})
        key, p0 = sample_rhos(key, psrs, {**gibbs_sites, **hmc_sites}, cs, invhdorf=invhdorf)
        return p0

    hmc_kernel = infer.NUTS(hmc_gibbs_model, **nuts_kwargs)
    kernel = infer.HMCGibbs(hmc_kernel, gibbs_fn=gibbs_step, gibbs_sites=gibbs_params)
    return kernel

