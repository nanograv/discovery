import time
import collections.abc
import typing
import warnings
PyTree = typing.Any

import numpy as np
import matplotlib.pyplot as pp
import IPython.display
import tqdm

import jax
import jax.numpy as jnp

try:
    import equinox as eqx
    import optax
    import flowjax
    import paramax
except ModuleNotFoundError:
    warnings.warn("Please install discovery[flow].")

# modified from flowjax/train/losses.py

def ElboLoss(target, num_samples):
    nbatch  = num_samples
    vtarget = jax.vmap(target)

    def theloss(params, static, key, beta):
        dist = eqx.combine(params, static)
        samples, log_probs = dist.sample_and_log_prob(key, (nbatch,))
        return (log_probs - beta * vtarget(samples)).mean()

    return theloss


def value_and_grad_ElboLoss(target, num_samples):
    nbatch = num_samples

    def theloss(params, static, key, beta):
        dist = eqx.combine(params, static)
        sample, log_prob = dist.sample_and_log_prob(key)
        return (log_prob - beta * target(sample)).mean()

    def meanloss(params, static, key, beta):
        keys = jax.random.split(key, num_samples)
        vals, grad = eqx.filter_vmap(eqx.filter_value_and_grad(theloss), in_axes=(None,None,0,None))(params, static, keys, beta)

        return vals.mean(), jax.tree_util.tree_map(lambda array: jnp.mean(array, axis=0), grad)

    meanloss.value_and_grad = True

    return meanloss


# modified from flowjax/train/train_utils.py to take combined loss + gradient,
# and to average over repeated minibatches

def step(params: PyTree,
         static: PyTree,
         key: jax.Array,
         beta: jax.Array,
         optimizer: optax.GradientTransformation,
         opt_state: PyTree,
         gradloss_fn: collections.abc.Callable,
         multibatch: int):
    """Carry out a training step.

    Args:
        params:      parameters for the model
        static:      static components of the model
        key:         JAX key passed to loss function
        beta:        inverse temperature passed to loss function
        optimizer:   Optax optimizer
        opt_state:   optimizer state
        gradloss_fn: loss and its gradient, e.g., from eqx.filter_value_and_grad(loss_fn)
        multibatch:  number of minibatches in this batch

    Returns:
        tuple:       (params, opt_state, loss_val)
    """
    losses, grads = [], []
    for i in range(multibatch):
        key, subkey = jax.random.split(key)
        loss, grad = gradloss_fn(params, static, subkey, beta)
        losses.append(loss)
        grads.append(grad)

    loss = np.mean(losses)
    grads = jax.tree_util.tree_map(lambda *x: sum(x) / multibatch, *grads)

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


# modified from flowjax/train/variational_fit.py

class VariationalFit:
    def __init__(self, dist, loss_fn, multibatch=1, optimizer=None, learning_rate=5e-4, annealing_schedule=None, patience=100, show_progress=True):
        """Set up a variational fit."""

        self.dist = dist
        # self.gradloss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn)) - see below
        self.multibatch = multibatch

        self.optimizer = optimizer or optax.adam(learning_rate)
        params, static = eqx.partition(self.dist, eqx.is_inexact_array, is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable))
        self.opt_state = self.optimizer.init(params)
        self.annealing_schedule = annealing_schedule or (lambda iter: 1.0)
        self.patience = patience

        # AOT: https://jax.readthedocs.io/en/latest/aot.html - seems to avoid recompilation
        key = jax.random.key(42)
        if hasattr(loss_fn, 'value_and_grad') and loss_fn.value_and_grad:
            self.gradloss_fn = eqx.filter_jit(loss_fn).lower(params, static, key, jnp.asarray(1.0)).compile()
        else:
            self.gradloss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn)).lower(params, static, key, jnp.asarray(1.0)).compile()

        self.show_progress = show_progress

        self.iter = 0
        self.losses = []
        self.best_params, self.best_iter = None, 0

    def run(self, key, steps = 100):
        params, static = eqx.partition(
            self.dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        if self.best_params:
            params = self.best_params

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, steps)

        if self.show_progress == True:
            keys = tqdm.tqdm(keys)

        for key in keys:
            params, self.opt_state, loss = step(params, static, key, beta=jnp.asarray(self.annealing_schedule(self.iter)),
                                                optimizer=self.optimizer, opt_state=self.opt_state,
                                                gradloss_fn=self.gradloss_fn, multibatch=self.multibatch)

            self.losses.append(loss.item())

            if loss.item() == min(self.losses):
                self.best_params = params
                self.best_iter = self.iter

            if self.iter - self.best_iter > self.patience:
                print(f'Early stopping at iteration {self.iter}')
                break

            if self.show_progress == True:
                keys.set_postfix({"loss": f'{self.losses[-1]:.2f}'})
            elif callable(self.show_progress):
                self.show_progress(eqx.combine(self.best_params, static), self.losses[-1], self.annealing_schedule(self.iter))

            self.iter = self.iter + 1

        if callable(self.show_progress) and hasattr(self.show_progress, 'freeze'):
            self.show_progress.freeze()

        return key, eqx.combine(self.best_params, static)


def hellinger(c1, c2, bins=100, range=None):
    """Hellinger distance of two vectors of variates."""

    if range is None:
        range = (np.min(c1), np.max(c1))

    h1, bn = np.histogram(c1, bins=bins, range=range, density=True)
    h2, _  = np.histogram(c2, bins=bins, range=range, density=True)

    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)

    return 1.0 - np.sum(np.sqrt(h1 * h2))


class display_flow:
    def __init__(self, logx, cadence=50, refchain=None, vlogx=None, save=None):
        self.logx = logx
        self.cadence, self.refchain, self.vlogx, self.save = cadence, refchain, vlogx, save

        self.t0, self.frozen = time.time(), None
        self.losses = []

    def freeze(self):
        self.frozen = time.time() - self.t0

    def thaw(self):
        if self.frozen is not None:
            self.t0 = time.time() - self.frozen
            self.frozen = None

    def __call__(self, flow, loss, temp):
        self.thaw()
        self.losses.append(loss)
        iteration = len(self.losses) - 1

        if callable(self.cadence):
            if not self.cadence(iteration):
                return
        else:
            if iteration % self.cadence != 0:
                return

        # ps = logx.to_df(flow.sample(train_key, sample_shape=(2*4096,)))

        train_key = jax.random.PRNGKey(42)

        if self.vlogx is None:
            ps = self.logx.to_df(flow.sample(train_key, sample_shape=(2*4096,)))
        else:
            samples, log_probs = flow.sample_and_log_prob(train_key, (4096,))
            ps = self.logx.to_df(samples)
            logweights = np.concatenate([
                self.vlogx(samples[i*512:(i+1)*512, :]) - log_probs[i*512:(i+1)*512]
                for i in range(8)
            ])
            weights = np.exp(logweights - np.max(logweights))

        if callable(self.save):
            self.save(iteration, ps)

        p1, p2 = self.logx.params[-2:]
        h1, h2 = 0.0, 0.0

        pp.clf()
        pp.gcf().set_figwidth(10)
        pp.gcf().set_figheight(2.5)
        try:
            pp.subplot(1, 4, 1)
            pp.plot(np.array(self.losses[-200:]) - self.losses[-1])
            pp.xlabel('iter (last 200)')
            pp.ylabel('loss')

            pp.subplot(1, 4, 2)
            pp.plot(ps[p1], ps[p2], '.', ms=1.0)
            pp.xlabel(p1)
            pp.ylabel(p2)

            pp.subplot(1, 4, 3)
            pp.hist(np.array(ps[p1]), histtype='step', density=True, bins=40)
            pp.xlabel(p1)

            if self.refchain is not None:
                pp.hist(np.array(self.refchain[p1]), histtype='step', density=True, bins=40)
                h1 = hellinger(self.refchain[p1], ps[p1])

            if self.vlogx is not None:
                pp.hist(ps[p1], histtype='step', density=True, bins=40, weights=weights)

            pp.subplot(1, 4, 4)
            pp.hist(np.array(ps[p2]), histtype='step', density=True, bins=40, label='nf')
            pp.xlabel(p2)

            if self.refchain is not None:
                pp.hist(np.array(self.refchain[p2]), histtype='step', density=True, bins=40, label='ref')
                h2 = hellinger(self.refchain[p2], ps[p2])

            if self.vlogx is not None:
                pp.hist(ps[p2], histtype='step', density=True, bins=40, weights=weights, label='rw')

            pp.legend()
        except:
            pass

        pp.suptitle(f'Iteration {iteration}, time = {(time.time() - self.t0):.0f}, loss = {self.losses[-1]:.2f}, beta = {temp:.2f}, hdist = {h1:.3f}, {h2:.3f}')
        pp.tight_layout()

        IPython.display.display(pp.gcf())
        IPython.display.clear_output(wait=True)
