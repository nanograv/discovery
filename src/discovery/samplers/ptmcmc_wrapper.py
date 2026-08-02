import os, jax
import numpy as np

import discovery as ds
from PTMCMCSampler import PTMCMCSampler as ptmcmc


class JumpProposal(object):

    def __init__(self, param_names, empirical_distr=None,
                 save_ext_dists=False, outdir='./chains'):
        """Set up some custom jump proposals"""

        self.params = param_names

        # parameter map
        self.pmap = {}
        ct = 0
        for p in self.params:
            self.pmap[str(p)] = slice(ct, ct+1)
            ct += 1

    def draw_from_prior(self, x, iter, beta):
        """Draw a sample from the prior distribution.
        The function signature is specific to PTMCMCSampler.

        Parameters:
        - x: The current state of the chain.
        - iter: The current iteration number.
        - beta: The current inverse temperature.

        Returns:
        - q: The new state drawn from the prior distribution.
        - lqxy: The log probability ratio of the forward-backward jump.

        """

        q = x.copy()
        lqxy = 0

        # randomly choose parameter
        param = np.random.choice(self.params)

        q[self.pmap[str(param)]] = list(ds.prior.sample_uniform([param]).values())

        # forward-backward jump probability
        lqxy += 0

        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        """
        Draw a sample from the red noise prior distribution.

        Parameters:
        - x: numpy.ndarray
            The current state of the parameters.
        - iter: int
            The current iteration number.
        - beta: float
            The inverse temperature parameter.

        Returns:
        - q: numpy.ndarray
            The new state of the parameters after drawing from the red noise prior.
        - lqxy: float
            The log of the forward-backward jump probability ratio.

        """

        q = x.copy()
        lqxy = 0

        signal_name = 'red_noise'
        red_pars = [p for p in self.params if signal_name in p]

        # draw parameter from signal model
        param = np.random.choice(red_pars)
        q[self.pmap[str(param)]] = list(ds.prior.sample_uniform([param]).values())

        # forward-backward jump probability
        lqxy += 0

        return q, float(lqxy)

    def draw_from_gwb_log_uniform_distribution(self, x, iter, beta):
        """
        Draws a sample from the log-uniform distribution for the GWB.

        Parameters:
        - x: The current state of the parameters.
        - iter: The current iteration number.
        - beta: The inverse temperature parameter.

        Returns:
        - q: The new state of the parameters after the draw.
        - lqxy: The log of the forward-backward jump probability ratio.
        """

        q = x.copy()
        lqxy = 0

        # draw parameter from signal model
        param = [p for p in self.params
                 if ('gw' in p and 'log10_A' in p)][0]

        q[self.pmap[str(param)]] = list(ds.prior.sample_uniform([param]).values())

        # forward-backward jump probability
        lqxy += 0

        return q, float(lqxy)


class InferenceModel(object):
    """
    A class representing an inference model.
    Parameters:
    - model: The model object.
    - pnames: A list of parameter names.

    Methods:
    - __init__(self, model, pnames=None): Initializes the InferenceModel object.
    - x2p(self, x): Converts a list of values to a dictionary of parameter-value pairs.
    - p2x(self, p): Converts a dictionary of parameter-value pairs to a list of values.
    - get_parameter_groups(self): Utility function to get parameter groupings for sampling.
    - setup_sampler(self, outdir='chains', resume=False, empirical_distr=None, groups=None, loglkwargs={}, logpkwargs={}): Sets up the sampler for MCMC sampling.
    """

    def __init__(self, model, pnames=None):
        """
        Initializes an instance of the `disco_ptmcmc` class.
        Parameters:
        - model: The model object.
        - pnames: A list of pulsar names.
        Returns:
        None
        """

        self.pnames = pnames
        self.param_names = model.logL.params

        loglike = model.logL
        logprior = ds.prior.makelogprior_uniform(self.param_names,
                                                 ds.prior.priordict_standard)

        jlogl = jax.jit(loglike)
        jlogp = jax.jit(logprior)

        self.get_lnlikelihood = lambda x: float(jlogl(self.x2p(x)))
        self.get_lnprior = lambda x: float(jlogp(self.x2p(x)))

    def x2p(self, x):
        """
        Converts a list of parameter values `x` to a dictionary representation.

        Args:
            x (list): A list of parameter values.

        Returns:
            dict: A dictionary representation of the parameter values, where the keys are the parameter names and the values are the corresponding values from `x`.
        """
        # does not handle vector parameters
        return {par: val for par, val in zip(self.param_names, x)}

    def p2x(self, p):
        """
        Convert a dictionary of values to a NumPy array.

        Parameters:
            p (dict): A dictionary containing values.

        Returns:
            numpy.ndarray: A NumPy array containing the values from the dictionary.
        """
        return np.array(list(p.values()), 'd')

    def get_parameter_groups(self):
        """Utility function to get parameter groupings for sampling."""
        params = self.param_names
        ndim = len(params)
        groups = [list(np.arange(0, ndim))]

        # get global and individual parameters
        gpars = [p for p in params if 'gw' in p or 'curn' in p]
        ipars = [p for p in params if p not in gpars]
        if gpars:
            # add a group of all global parameters
            groups.append([params.index(gp) for gp in gpars])
        if ipars:
            #groups.append([params.index(ip) for ip in ipars])
            groups += [[params.index(ip) for ip in ipars if p in ip]
                       for p in self.pnames]

        return groups

    def setup_sampler(self, outdir='chains', resume=False,
                      empirical_distr=None, groups=None,
                      loglkwargs={}, logpkwargs={}):
        """
        Set up the sampler for performing MCMC (Markov Chain Monte Carlo) sampling.

        Parameters:
        - outdir (str): The output directory for saving the chains. Default is 'chains'.
        - resume (bool): Whether to resume from a previous run. Default is False.
        - empirical_distr (None or str): The path to the empirical distribution file. Default is None.
        - groups (None or list): The parameter groups for the sampler. Default is None.
        - loglkwargs (dict): Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        - logpkwargs (dict): Additional keyword arguments for the log-prior function. Default is an empty dictionary.

        Returns:
        - sampler: The initialized PTMCMCSampler object.
        """

        # dimension of parameter space
        ndim = len(self.param_names)

        # initial jump covariance matrix
        if os.path.exists(outdir+'/cov.npy') and resume:
            cov = np.load(outdir+'/cov.npy')

            # check that the one we load is the same shape as our data
            cov_new = np.diag(np.ones(ndim) * 1.0**2)
            if cov.shape != cov_new.shape:
                msg = 'The covariance matrix (cov.npy) in the output folder is '
                msg += 'the wrong shape for the parameters given. '
                msg += 'Start with a different output directory or '
                msg += 'change resume to False to overwrite the run that exists.'

                raise ValueError(msg)
        else:
            cov = np.diag(np.ones(ndim) * 1.0**2)  # used to be 0.1

        # parameter groupings
        if groups is None:
            groups = self.get_parameter_groups()

        sampler = ptmcmc.PTSampler(ndim, self.get_lnlikelihood, self.get_lnprior, cov,
                                   groups=groups, outDir=outdir, resume=resume,
                                   loglkwargs=loglkwargs, logpkwargs=logpkwargs)

        # additional jump proposals
        jp = JumpProposal(param_names=self.param_names, empirical_distr=None,
                          save_ext_dists=False, outdir=outdir)
        sampler.jp = jp

        # always add draw from prior
        sampler.addProposalToCycle(jp.draw_from_prior, 5)

        # try adding empirical proposals
        #if empirical_distr is not None:
        #    print('Adding empirical proposals...\n')
        #    sampler.addProposalToCycle(jp.draw_from_empirical_distr, 25)

        # Red noise prior draw
        if any('red_noise' in s for s in self.param_names):
            print('Adding red noise prior draws...\n')
            sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

        # GWB uniform distribution draw
        if np.any([('gw' in par and 'log10_A' in par) for par in self.param_names]):
            print('Adding GWB uniform distribution draws...\n')
            sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

        # free spectrum prior draw
        #if np.any(['log10_rho' in par for par in self.param_names]):
        #    print('Adding free spectrum prior draws...\n')
        #    sampler.addProposalToCycle(jp.draw_from_gw_rho_prior, 25)

        # Prior distribution draw for parameters named GW
        #if any([str(p).split(':')[0] for p in list(self.params) if 'gw' in str(p)]):
        #    print('Adding gw param prior draws...\n')
        #    sampler.addProposalToCycle(jp.draw_from_par_prior(
        #        par_names=[str(p).split(':')[0] for
        #                   p in list(self.params)
        #                  if 'gw' in str(p)]), 10)

        return sampler