import pandas as pd
import jax
from jax import numpy as jnp
from typing import Callable


def dict_batches(model1_df: pd.DataFrame, batch_size: int):
    """
    Yield sub-dicts of size batch_size along the first axis
    of every array in data_dict that shares the full data length.
    Keys whose arrays don't match that length are passed through intact.
    """
    data_dict = {col: jnp.array(model1_df[col].values) for col in model1_df.columns}
    total = next(iter(data_dict.values())).shape[0]   # length of first array

    for start in range(0, total, batch_size):
        end = start + batch_size
        def slice_fn(x):
            if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] == total:
                return x[start:end]
            else:
                return x
        yield jax.tree_util.tree_map(slice_fn, data_dict)

def batch_reweight(model1_df: pd.DataFrame, model2_logl: Callable, batch_size=64):
    batches = dict_batches(model1_df, batch_size)
    jvlogl = jax.jit(jax.vmap(model2_logl))

    log_likelihoods = []
    for batch in batches:
        log_likelihoods.append(jvlogl(batch))
    model2_df = pd.DataFrame(model1_df)
    model2_df['logl'] = jnp.concatenate(log_likelihoods)
    return model2_df

def compute_weights(model1_df, model2_df):
    """
    Compute weights for model2_df based on the log-likelihoods of model1_df.
    """
    logl1 = model1_df['logl'].values
    logl2 = model2_df['logl'].values

    # Compute weights
    weights = jnp.exp(logl2 - logl1)
    return weights

def compute_bayes_factor(model1_df, model2_df):
    weights = compute_weights(model1_df, model2_df)
    bayes_factor = jnp.average(weights)
    bayes_factor_unc = jnp.std(weights) / jnp.sqrt(len(weights))
    return bayes_factor, bayes_factor_unc
