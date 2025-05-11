import inspect
import warnings

import pandas as pd

try:
    import numpyro
    from numpyro import infer
    from numpyro import distributions as dist
except ModuleNotFoundError:
    warnings.warn("NumPyro not found!")


from discovery import prior


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(-10,10).expand([parlen]))
        logl = logx(pars)

        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: logx.to_df(chain['pars'])

    return numpyro_model


def makemodel(mylogl, priordict={}):
    def numpyro_model():
        logl = mylogl({par: numpyro.sample(par, dist.Uniform(*prior.getprior_uniform(par, priordict)))
                       for par in mylogl.params})

        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model


def makesampler_nuts(numpyro_model, num_warmup=512, num_samples=1024, num_chains=1, **kwargs):
    nutsargs = dict(max_tree_depth=8, dense_mass=False,
                    forward_mode_differentiation=False, target_accept_prob=0.8,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.NUTS).args})

    mcmcargs = dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method='vectorized', progress_bar=True,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs})

    sampler = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)
    sampler.to_df = lambda: numpyro_model.to_df(sampler.get_samples())

    return sampler
