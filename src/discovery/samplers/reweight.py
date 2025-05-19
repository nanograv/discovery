import pandas as pd
import jax
from jax import numpy as jnp
from typing import Callable


def batch_reweight(
    source_df: pd.DataFrame,
    target_logl_fn: Callable,
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Compute the log-likelihood of each sample in source_df under a new model
    (given by target_logl_fn) in batches, and return a copy of source_df
    augmented with a 'logl' column of the recomputed values.

    Args:
        source_df: DataFrame of samples (one row per sample, columns = parameters).
                   If it contains a 'logl' column it will be dropped.
        target_logl_fn: Function that maps a 1D array of parameter values to a scalar
                        log-likelihood.
        batch_size: Number of samples to process per vmapped function call.

    Returns:
        DataFrame: A copy of source_df with 'logl' from target_logl_fn.
    """
    df = source_df.copy()
    if 'logl' in df.columns:
        df = df.drop(columns=['logl'])
    param_array = jnp.array(df.to_numpy())

    jitted_logl_fn = jax.jit(target_logl_fn)
    recomputed_logl = jax.lax.map(jitted_logl_fn, param_array, batch_size=batch_size)

    result_df = pd.DataFrame(df)
    result_df['logl'] = recomputed_logl
    return result_df

def compute_weights(
    base_df: pd.DataFrame,
    reweighted_df: pd.DataFrame
) -> jnp.ndarray:
    """
    Given two DataFrames with 'logl' columns, compute importance weights w_i = exp(logl2_i - logl1_i).

    Args:
        base_df: Original samples with their log-likelihoods under the first model.
        reweighted_df: Same samples with log-likelihoods under the second model.

    Returns:
        Array of weights of shape (N,).
    """
    logl1 = base_df['logl'].to_numpy()
    logl2 = reweighted_df['logl'].to_numpy()
    return jnp.exp(logl2 - logl1)

def compute_bayes_factor(
    base_df: pd.DataFrame,
    reweighted_df: pd.DataFrame
) -> tuple[float, float]:
    """
    Estimate the Bayes factor between two models:
        BF = E[w] = mean_i exp(logl2_i - logl1_i),
    with uncertainty σ_w / sqrt(N).

    Args:
        base_df: DataFrame of samples from model 1 with 'logl' column.
        reweighted_df: DataFrame of same samples under model 2 with 'logl' column.

    Returns:
        A tuple (bayes_factor, uncertainty).
    """
    weights = compute_weights(base_df, reweighted_df)
    bf = jnp.mean(weights)
    bf_unc = jnp.std(weights) / jnp.sqrt(len(weights))
    return float(bf), float(bf_unc)
