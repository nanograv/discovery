import inspect
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd

import numpyro
from numpyro import infer
from numpyro import distributions as dist

from .. import prior
from ..pulsar import save_chain


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))
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

def run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint,
    rng_key,
    outdir="chains",
    resume=False,
):
    """Run NumPyro MCMC and save checkpoints.

    This function performs multiple iterations of MCMC sampling, saving checkpoints
    after each iteration. It saves samples to feather files and the NumPyro MCMC
    state to JSON.

    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        A NumPyro MCMC sampler object.
    num_samples_per_checkpoint : int
        The number of samples to save in each checkpoint.
    rng_key : jax.random.PRNGKey
        The random number generator key for JAX.
    outdir : str | Path
        The directory for output files.
    resume : bool
        Whether to look for a state to resume from.

    Returns
    -------
    None
        This function doesn't return any value but saves the results to disk.

    Side Effects
    ------------
    - Runs the MCMC sampler for the specified number of `num_sampling_iterations`.
    - Saves samples data to feather files after each iteration.
    - Writes the NumPyro sampler state to a JSON file after each iteration.

    Example
    -------
    >>> import discovery.samplers.numpyro as ds_numpyro
    >>> # Assume `model` is configured
    >>> npsampler = ds_numpyro.makesampler_nuts(model, num_samples =100, num_warmup=50)
    >>> ds_numpyro.run_nuts_with_checkpoints(npsampler, 10, jax.random.key(42))

    """
    # convert to pathlib object
    # make directory if it doesn't exist
    if not isinstance(outdir, Path):
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

    samples_file = outdir / "numpyro-samples.feather"
    checkpoint_file = outdir / "numpyro-checkpoint.pickle"

    if checkpoint_file.is_file() and samples_file.is_file() and resume:
        df = pd.read_feather(samples_file)
        num_samples_saved = df.shape[0]

        with checkpoint_file.open("rb") as f:
            checkpoint = pickle.load(f)

        total_sample_num = sampler.num_samples - num_samples_saved

        sampler.post_warmup_state = checkpoint

    else:
        df = None
        num_samples_saved = 0
        total_sample_num = sampler.num_samples

    num_checkpoints = int(jnp.ceil(total_sample_num / num_samples_per_checkpoint))
    remainder_samples = int(total_sample_num % num_samples_per_checkpoint)

    for checkpoint in range(num_checkpoints):
        if checkpoint == 0:
            sampler.num_samples = num_samples_per_checkpoint
            sampler._set_collection_params()  # Need this to update num_samples
        elif checkpoint == num_checkpoints - 1:
            # We won't need to update the collection params because we've set the post warmup state,
            # and that accomplishes the same goal.
            sampler.num_samples = remainder_samples if remainder_samples != 0 else num_samples_per_checkpoint

        sampler.run(rng_key)

        df_new = sampler.to_df()

        df = pd.concat([df, df_new]) if df is not None else df_new

        save_chain(df, samples_file)

        with checkpoint_file.open("wb") as f:
            pickle.dump(sampler.last_state, f)

        sampler.post_warmup_state = sampler.last_state

        rng_key, _ = jax.random.split(rng_key)
