import jax
import optax 
import equinox as eqx
import numpy as np
import json
import jax.random as jr
import prior

from flowjax.bijections import RationalQuadraticSpline, Affine
from flowjax.distributions import StandardNormal, Transformed
from flowjax.flows import masked_autoregressive_flow


# build the flow
def make(*, key, loglike, flow_arch = masked_autoregressive_flow, base_distribution = StandardNormal, transform = RationalQuadraticSpline,
          n_samples = 512, knots = 8, interval = 5, patience = 100, multibatch = 1, LR = 1e-3, steps = 2000, flow_lay = 8, deepness = 1):
        
        l = np.array([prior.sample_uniform(loglike.logL.params)[k] for k in prior.sample_uniform(loglike.logL.params).keys()])

        key, flow_key, train_key = jr.split(key, 3)
        flow = flow_arch(flow_key, base_dist= base_distribution(l.shape), flow_layers = flow_lay, nn_depth = deepness, transformer=transform(knots = knots, interval=interval), invert=False)

        return flow

# save the hyperparameters
def save(filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)

# load the model
def load(filename, loglike, flow_arch = masked_autoregressive_flow, base_distribution = StandardNormal, transform = RationalQuadraticSpline):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            # in case I set an interval instead of a single number
            if isinstance(hyperparams['interval'], list):
                hyperparams['interval'] = (hyperparams['interval'][0], hyperparams['interval'][1])

            model = make(key=jr.PRNGKey(42), loglike = loglike, flow_arch = flow_arch, base_distribution = base_distribution, transform = transform, **hyperparams)
            print(hyperparams)
            return eqx.tree_deserialise_leaves(f, model)
