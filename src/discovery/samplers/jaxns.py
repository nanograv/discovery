import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

import numpyro.distributions as dist
from jaxns import Model, Prior, NestedSampler, resample
from .. import prior

def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}): # for drop-in compatibility with numpyro

    return makemodel(mylogl, priordict)


def makemodel(mylogl, priordict={}):
    params = list(mylogl.params)

    def prior_model():
        values = []
        for par in params:
            low, high = prior.getprior_uniform(par, priordict)
            if '(' in par:
                base = par.split('(')[0]
                size = int(par[par.index('(') + 1 : par.index(')')])
                low_arr = jnp.full((size,), low)
                high_arr = jnp.full((size,), high)
                val = yield Prior(tfpd.Uniform(low=low_arr, high=high_arr), name=base)
            else:
                val = yield Prior(tfpd.Uniform(low=low, high=high), name=par)
            values.append(val)
        if len(values) == 1:
            return values[0]

        return tuple(values)

    def log_likelihood(*args):
        params_dict = {}
        i = 0
        for par in params:
            if '(' in par:
                val = args[i]
                params_dict[par] = val
                i += 1
            else:
                val = args[i]
                params_dict[par] = val
                i += 1
        return mylogl(params_dict)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model


def makesampler_nested(model, max_samples=1e6, num_live_points=None, **kwargs):

    ns = NestedSampler(model=model, parameter_estimation=True, max_samples=max_samples, num_live_points=num_live_points, verbose=True, **kwargs)

    class Sampler:
        def __init__(self, nested_sampler):
            self.ns = nested_sampler

        def run(self, key):
            self.termination, self.state = self.ns(key)
            self.results = self.ns.to_results(termination_reason=self.termination, state=self.state)
            self.ns.summary(self.results)

        def make_plots(self, save_name=None):
            self.ns.plot_diagnostics(self.results, save_name=save_name+'_diagnostics.png')
            self.ns.plot_cornerplot(self.results, save_name=save_name+'_corner.png')

        def to_df(self, S=None, seed=0):
            samples = self.results.samples
            log_weights = self.results.log_dp_mean
            if S is None:
                first_key = next(iter(samples))
                S = samples[first_key].shape[0]
            resampled = resample(jax.random.PRNGKey(seed), samples=samples, log_weights=log_weights, S=S)
            data = {}
            for name, arr in resampled.items():
                arr_np = np.array(arr)
                if arr_np.ndim == 1:
                    data[name] = arr_np
                else:
                    for j in range(arr_np.shape[1]):
                        data[f"{name}[{j}]"] = arr_np[:, j]

            return pd.DataFrame(data)

    return Sampler(ns)
