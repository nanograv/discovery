Conditional Sampling of GP Coefficients
========================================

Discovery provides methods to sample GP coefficients conditionally on hyperparameter values.
This allows drawing realizations of stochastic processes given their amplitude and spectral
properties, which is useful for simulations and diagnostics.

Overview
--------

In Discovery's GP framework (see :doc:`/guide/data_model`), a Gaussian process is represented by:

- **Basis matrix** :math:`F` (``ntoas × ngp``)
- **Coefficients** :math:`a` (latent variables)
- **Prior covariance** :math:`\Phi(\Lambda)` determined by hyperparameters :math:`\Lambda`

The marginalized likelihood integrates out the coefficients :math:`a`. However, we can
sample from the conditional distribution:

.. math::

   p(a | y, \Lambda) \propto p(y | a) \, p(a | \Lambda)

This gives us realizations of the GP coefficients that are consistent with both the data
and the hyperparameters.

Conditional Distribution
------------------------

Given data :math:`y`, noise :math:`N`, basis :math:`F`, and hyperparameters :math:`\Lambda`,
the conditional posterior for coefficients is Gaussian:

.. math::

   a | y, \Lambda \sim \mathcal{N}(\mu_a, \Sigma_a)

where:

.. math::

   \Sigma_a &= (\Phi^{-1} + F^T N^{-1} F)^{-1} \\
   \mu_a &= \Sigma_a F^T N^{-1} y

The mean :math:`\mu_a` represents the maximum a posteriori (MAP) estimate of the coefficients,
while :math:`\Sigma_a` quantifies the uncertainty given the data and hyperparameters.

Single Pulsar
-------------

The ``conditional`` Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``conditional`` method to get the mean and covariance:

.. code-block:: python

   import discovery as ds

   # Build likelihood
   psr = ds.Pulsar.read_feather('data/v1p1_de440_pint_bipm2019-B1855+09.feather')

   signals = [
       psr.residuals,
       ds.makenoise_measurement(psr, noisedict, ecorr=True),
       ds.makegp_timing(psr, svd=True),
       ds.makegp_fourier(psr, ds.powerlaw, 30, name='rn')
   ]

   logl = ds.PulsarLikelihood(signals)

   # Sample hyperparameters from priors
   params = ds.sample_uniform(logl.params, priordict)

   # Get conditional distribution
   mu, cf = logl.conditional(params)

**Returns:**

- ``mu``: Mean of conditional distribution (concatenated coefficients for all GPs)
- ``cf``: Cholesky factor of covariance matrix :math:`\Sigma_a`

The covariance can be reconstructed as :math:`\Sigma_a = L L^T` where ``L = cf[0]``.

**Note:** :func:`~discovery.signals.makegp_timing` does not produce coefficients in the
conditional distribution (it uses an improper prior by default).

See :func:`~discovery.likelihood.PulsarLikelihood.conditional`.

Sampling from the Conditional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw samples from the conditional distribution:

.. code-block:: python

   import jax

   # Get conditional sampler
   sampler = logl.sample_conditional

   # Draw a sample
   key = jax.random.PRNGKey(42)
   key, coefficients = sampler(key, params)

**Returns:**

- ``key``: Updated PRNG key for subsequent sampling
- ``coefficients``: Dictionary with coefficient arrays for each GP

**Example coefficient dictionary:**

.. code-block:: python

   {
       'B1855+09_rn_coefficients': array([...]),  # shape (60,) for 30 components
   }

See :func:`~discovery.likelihood.PulsarLikelihood.sample_conditional`.

Multiple Pulsars (GlobalLikelihood)
-----------------------------------

For multi-pulsar analyses, use :class:`~discovery.likelihood.GlobalLikelihood`:

.. code-block:: python

   import discovery as ds
   import glob

   # Load pulsars
   files = sorted(glob.glob('data/v1p1_de440_pint_bipm2019-*.feather'))
   psrs = [ds.Pulsar.read_feather(f) for f in files]
   Tspan = ds.getspan(psrs)

   # Build per-pulsar likelihoods with common processes
   pulsarlikelihoods = []
   for psr in psrs:
       signals = [
           psr.residuals,
           ds.makenoise_measurement(psr, noisedict, ecorr=True),
           ds.makegp_timing(psr, svd=True),
           ds.makegp_fourier(psr, ds.powerlaw, 30, name='rn')
       ]
       pulsarlikelihoods.append(ds.PulsarLikelihood(signals))

   # Build global likelihood with correlated GW signal
   gbl = ds.GlobalLikelihood(
       pulsarlikelihoods,
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan, name='gw'
       )
   )

   # Sample hyperparameters from priors
   params = ds.sample_uniform(gbl.logL.params, priordict)

   # Sample conditional coefficients
   key = jax.random.PRNGKey(0)
   key, coefficients = gbl.sample_conditional(key, params)

**Coefficient structure:**

For :class:`~discovery.likelihood.GlobalLikelihood`, the coefficients dictionary contains
separate entries for each pulsar's red noise GP and for each pulsar's contribution to the
global GW signal:

.. code-block:: python

   {
       'B1855+09_rn_coefficients': array([...]),  # shape (60,) - pulsar 1 red noise
       'B1937+21_rn_coefficients': array([...]),  # shape (60,) - pulsar 2 red noise
       # ... other pulsars' red noise
       'B1855+09_gw_coefficients': array([...]),  # shape (28,) - pulsar 1 GW contribution
       'B1937+21_gw_coefficients': array([...]),  # shape (28,) - pulsar 2 GW contribution
       # ... other pulsars' GW contributions
   }

Each pulsar has separate coefficient arrays for its intrinsic processes (red noise) and its
contribution to the global correlated signal (GW).

See :func:`~discovery.likelihood.GlobalLikelihood.sample_conditional`.

Multiple Samples
~~~~~~~~~~~~~~~~

To draw multiple independent samples:

.. code-block:: python

   # Draw 100 samples
   key = jax.random.PRNGKey(0)
   samples = []

   for _ in range(100):
       key, coeffs = gbl.sample_conditional(key, params)
       samples.append(coeffs)

**Note:** Each call updates the PRNG key to ensure independent samples.

Use Cases
---------

Conditional sampling is used in several analysis techniques:

- **Outlier identification:** Used to identify timing outliers by examining residuals after
  removing the conditional GP realization. See `Wang and Taylor 2021 <https://arxiv.org/abs/2112.05698>`_.

- **Signal reconstruction and posterior predictive tests:** Used to reconstruct stochastic
  signals and perform posterior predictive checks. See `NANOGrav 2024 <https://arxiv.org/abs/2407.20510>`_.

Limitations
-----------

**Deterministic Delays:**

Currently, conditional sampling does not support models with deterministic delays:

.. code-block:: python

   # This will raise NotImplementedError
   signals = [
       psr.residuals,
       ds.makenoise_measurement(psr, noisedict, ecorr=True),
       ds.makegp_timing(psr, svd=True),
       ds.makegp_fourier(psr, ds.powerlaw, 30, name='rn'),
       ds.makedelay(psr, delay_func, name='delay')  # Not supported
   ]

   logl = ds.PulsarLikelihood(signals)
   # logl.conditional(params)  # raises NotImplementedError

**Workaround:** Build a separate likelihood without delays for conditional sampling.

See Also
--------

- :doc:`/guide/data_model` - GP mathematical framework
- :doc:`/tutorials/simulations` - Data simulation workflows
- :class:`~discovery.likelihood.PulsarLikelihood` - Likelihood API
- :class:`~discovery.likelihood.GlobalLikelihood` - Multi-pulsar likelihood API
- :func:`~discovery.likelihood.PulsarLikelihood.conditional` - Get conditional parameters
- :func:`~discovery.likelihood.PulsarLikelihood.sample_conditional` - Sample from conditional
- :func:`~discovery.prior.sample_uniform` - Sample from uniform priors
