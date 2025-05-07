import re

import numpy as np
import pandas as pd

from discovery import matrix
jnp = matrix.jnp

def uniform(par, a, b):
    def logpriorfunc(params):
        return matrix.jnp.where(matrix.jnp.logical_and(params[par] >= a, params[par] <= b), 0, -matrix.jnp.inf)

    return logpriorfunc


priordict_standard = {
    "(.*_)?efac": [0.9, 1.1],
    "(.*_)?t2equad": [-8.5, -5],
    "(.*_)?tnequad": [-8.5, -5],
    "(.*_)?log10_ecorr": [-8.5, -5],
    "(.*_)?rednoise_log10_A.*": [-20, -11],
    "(.*_)?rednoise_gamma.*": [0, 7],
    "(.*_)?rednoise_log10_fb": [-9, -6],
    "(.*_)?red_noise_log10_A.*": [-20, -11],  # deprecated
    "(.*_)?red_noise_gamma.*": [0, 7],  # deprecated
    "(.*_)?red_noise_log10_fb": [-9, -6],
    "crn_log10_A.*": [-18, -11],
    "crn_gamma.*": [0, 7],
    "crn_log10_fb": [-9, -6],
    "gw_(.*_)?log10_A": [-18, -11],
    "gw_(.*_)?gamma": [0, 7],
    "gw_log10_fb": [-9, -6],
    "(.*_)?dmgp_log10_A": [-20, -11],
    "(.*_)?dmgp_gamma": [0, 7],
    "(.*_)?dmgp_alpha": [1, 3],
    "crn_log10_rho": [-9, -4],
    "gw_(.*_)?log10_rho": [-9, -4],
    r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": [-9, -4],
    r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": [-9, -4]
}

def getprior_uniform(par, priordict={}):
    priordict = {**priordict_standard, **priordict}

    for parname, range in priordict.items():
        if re.match(parname, par):
            return range

    raise KeyError(f'getprior_uniform: no prior for parameter {par}.')

def makelogprior_uniform(params, priordict={}):
    priordict = {**priordict_standard, **priordict}

    priors = []
    for par in params:
        for parname, range in priordict.items():
            if re.match(parname, par):
                priors.append(uniform(par, *range))
                break

    def logprior(params):
        return sum(prior(params) for prior in priors)

    return logprior


def makelogtransform_uniform(func, priordict={}):
    priordict = {**priordict_standard, **priordict}

    # figure out slices when there are vector arguments
    slices, offset = [], 0
    for par in func.params:
        if '(' in par:
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1
            slices.append(slice(offset, offset+l))
            offset = offset + l
        else:
            slices.append(offset)
            offset = offset + 1

    # build vectors of DF column names and of lower and upper uniform limits
    a, b = [], []
    columns = []
    for par, slice_ in zip(func.params, slices):
        for pname, prange in priordict.items():
            if re.match(pname, par):
                therange = prange
                break
        else:
            raise KeyError(f"No known prior for {par}.")

        if '(' in par:
            root = par[:par.index('(')]
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1

            for i in range(l):
                columns.append(f'{root}[{i}]')
                a.append(therange[0])
                b.append(therange[1])
        else:
            columns.append(par)
            a.append(therange[0])
            b.append(therange[1])

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))

        if len(a) != len(func.params):
            return {par: xs[slice_] for par, slice_ in zip(func.params, slices)}
        else:
            return dict(zip(func.params, xs))

    def to_vec(params):
        xs = jnp.zeros_like(a)
        for par, slice_ in zip(func.params, slices):
            xs = xs.at[slice_].set(params[par])

        # only Python 3.11: xs = jnp.r_[*[params[pname] for pname in func.params]]
        # only scalar parameters: xs = matrix.jnparray([params[pname] for pname in func.params])

        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys, psrs=None):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))

        if psrs is None:
            return pd.DataFrame(np.array(xs), columns=columns)
        else:
            # rename columns from psr number to psr name
            psrdict = {f'{i}]': psr.name for i, psr in enumerate(psrs)}
            psrcols = [psrdict[par.split('[')[1]] + '_' + par.split('[')[0] if '[' in par else par for par in columns]
            return pd.DataFrame(np.array(xs), columns=psrcols).sort_index(axis=1)

    def prior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

    def transformed(ys):
        return func(to_dict(ys)) + prior(ys)

    transformed.params = func.params

    transformed.prior = prior
    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df

    return transformed


def makelogtransform_classic(func, priordict={}):
    priordict = {**priordict_standard, **priordict}

    a, b = [], []
    for par in func.params:
        for pname, prange in priordict.items():
            if re.match(pname, par):
                a.append(prange[0])
                b.append(prange[1])
                break
        else:
            raise KeyError(f"No known prior for {par}.")

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return dict(zip(func.params, xs))

    def to_vec(params):
        xs = matrix.jnparray([params[pname] for pname in func.params])
        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return pd.DataFrame(np.array(xs), columns=func.params)

    def prior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

        # return jnp.sum(jnp.log(0.5) - 2.0 * jnp.log(jnp.cosh(ys)))
        # but   log(0.5) - 2 * log(cosh(y))
        #     = log(0.5) - 2 * log((exp(x) + exp(-x))/2)
        #     = log(0.5) - 2 * (log(exp(x) - exp(-x)) - log(2.0))
        #     = log(2.0) - 2 * logaddexp(x, -x)

    def transformed(ys):
        return func(to_dict(ys)) + prior(ys)

    transformed.params = func.params

    transformed.prior = prior
    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df

    return transformed


def sample_uniform(params, priordict={}, n=1, fail=True):
    priordict = {**priordict_standard, **priordict}

    sample = {}
    for par in params:
        for parname, range in priordict.items():
            if parname == par or re.match(parname, par):
                if par.endswith(")"):
                    sample[par] = (
                        np.random.uniform(*range, size=int(par[par.index("(") + 1 : -1]))
                        if n == 1
                        else np.random.uniform(*range, size=(n, int(par[par.index("(") + 1 : -1])))
                    )
                else:
                    sample[par] = np.random.uniform(*range) if n == 1 else np.random.uniform(*range, size=n)
                break
        else:
            if fail:
                raise KeyError(f"No known prior for {par}.")

    return sample
