Basic Likelihood
================

This tutorial demonstrates how to build likelihoods in Discovery, from single pulsars to pulsar timing arrays.

Single Pulsar Likelihood
-------------------------

Loading Data
~~~~~~~~~~~~

First, load a pulsar dataset:

.. code-block:: python

   import discovery as ds

   psr = ds.Pulsar.read_feather('data/v1p1_de440_pint_bipm2019-B1855+09.feather')

Building the Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~

A basic likelihood combines residuals, noise, and signal components:

.. code-block:: python

   signals = [
       psr.residuals,                                  # Data vector
       ds.makenoise_measurement(psr, psr.noisedict),  # White noise (EFAC/EQUAD)
       ds.makegp_ecorr(psr, psr.noisedict),           # ECORR noise
       ds.makegp_timing(psr, svd=True),               # Timing model
       ds.makegp_fourier(psr, ds.powerlaw, 30,        # Red noise
                        name='rednoise')
   ]

   psl = ds.PulsarLikelihood(signals)
   logl = psl.logL

The resulting ``logl`` is a JAX-ready function that takes a parameter dictionary.

Understanding Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The likelihood automatically names parameters based on pulsar name and component:

.. code-block:: python

   print(psl.logL.params)
   # ['B1855+09_rednoise_log10_A', 'B1855+09_rednoise_gamma']

Parameters in ``noisedict`` are **fixed** and excluded from the parameter list:

.. code-block:: python

   # These are fixed (in psr.noisedict):
   # - B1855+09_430_ASP_efac, B1855+09_430_ASP_equad, etc.

   # These are free (not in noisedict):
   # - B1855+09_rednoise_log10_A
   # - B1855+09_rednoise_gamma

Evaluating the Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a parameter dictionary and evaluate:

.. code-block:: python

   import jax

   params = {
       'B1855+09_rednoise_log10_A': -14.0,
       'B1855+09_rednoise_gamma': 4.33
   }

   # JIT-compile for performance
   logl_jit = jax.jit(logl)
   logL_value = logl_jit(params)

Computing Gradients
~~~~~~~~~~~~~~~~~~~

JAX provides automatic differentiation:

.. code-block:: python

   # Gradient function
   grad_logl = jax.jit(jax.grad(logl))

   # Evaluate gradient
   grads = grad_logl(params)
   # {'B1855+09_rednoise_log10_A': ..., 'B1855+09_rednoise_gamma': ...}

Multi-Pulsar Analysis: GlobalLikelihood
----------------------------------------

For analyzing multiple pulsars with a correlated process (e.g., gravitational wave background),
use ``GlobalLikelihood``:

Loading Multiple Pulsars
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import glob

   psrs = [ds.Pulsar.read_feather(f)
           for f in glob.glob('data/v1p1_de440_pint_bipm2019-*.feather')]

   Tspan = ds.getspan(psrs)

Building Individual Likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``PulsarLikelihood`` for each pulsar:

.. code-block:: python

   pulsarlikelihoods = []
   for psr in psrs:
       psl = ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ])
       pulsarlikelihoods.append(psl)

Adding a Correlated Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a global likelihood with Hellings-Downs correlated GW background:

.. code-block:: python

   gbl = ds.GlobalLikelihood(
       pulsarlikelihoods,
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   logl = gbl.logL

Parameters now include both pulsar-specific and common terms:

.. code-block:: python

   print(gbl.logL.params)
   # ['B1855+09_rednoise_log10_A', 'B1855+09_rednoise_gamma',
   #  'B1937+21_rednoise_log10_A', 'B1937+21_rednoise_gamma',
   #  ...,
   #  'gw_log10_A', 'gw_gamma']

Multi-Pulsar Analysis: ArrayLikelihood
---------------------------------------

``ArrayLikelihood`` is optimized for **vectorized operations across pulsars**. When all pulsars
have the same noise model structure (same number of components, same prior forms),
ArrayLikelihood can batch operations for dramatic speedups, especially on GPUs.

Common Red Noise Model
~~~~~~~~~~~~~~~~~~~~~~~

First, create likelihoods with identical noise model structures:

.. code-block:: python

   pulsarlikelihoods = [
       ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True, constant=1e-6)
       ]) for psr in psrs
   ]

Now add a vectorized common red noise model:

.. code-block:: python

   curn = ds.ArrayLikelihood(
       pulsarlikelihoods,
       commongp=ds.makecommongp_fourier(
           psrs, ds.makepowerlaw_crn(14), 30, T=Tspan,
           common=['crn_log10_A', 'crn_gamma'],
           name='red_noise'
       )
   )

The :func:`~discovery.signals.makepowerlaw_crn` function combines intrinsic red noise
(with per-pulsar amplitude and spectral index) and a common process (with shared parameters).
This allows efficient vectorization: all pulsars use the same prior form, enabling batched
operations across the pulsar array.

GW Background with ArrayLikelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   hd = ds.ArrayLikelihood(
       pulsarlikelihoods,
       commongp=ds.makecommongp_fourier(
           psrs, ds.powerlaw, 30, T=Tspan,
           name='red_noise'
       ),
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   logl = hd.logL

This implements per-pulsar red noise (vectorized across pulsars) plus a GW background
with Hellings-Downs correlation.

Adding a Prior
--------------

Define priors for free parameters:

.. code-block:: python

   # Use Discovery's default priors
   logprior = ds.makelogprior_uniform(logl.params)

   # Or specify custom priors
   priordict = {
       'gw_log10_A': [-18, -11],
       'gw_gamma': [0, 7]
   }
   logprior = ds.makelogprior_uniform(logl.params, priordict)

   # JIT-compile
   logp = jax.jit(logprior)

Complete Example
----------------

Here's a complete multi-pulsar HD analysis:

.. code-block:: python

   import discovery as ds
   import jax
   import glob

   # Load data
   psrs = [ds.Pulsar.read_feather(f)
           for f in glob.glob('data/v1p1_de440_pint_bipm2019-*.feather')[:5]]
   Tspan = ds.getspan(psrs)

   # Build global likelihood with HD correlation
   gbl = ds.GlobalLikelihood(
       [ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ]) for psr in psrs],
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   # Define prior
   logprior = ds.makelogprior_uniform(gbl.logL.params)

   # JIT-compile
   logl = jax.jit(gbl.logL)
   logp = jax.jit(logprior)
   grad_logl = jax.jit(jax.grad(logl))

   # Sample from prior
   params = ds.sample_uniform(gbl.logL.params)

   # Evaluate
   print(f"Log-likelihood: {logl(params)}")
   print(f"Log-posterior: {logl(params) + logp(params)}")

Next Steps
----------

- :doc:`simulations` - Generating synthetic data
- :doc:`optimal_statistic` - Optimal statistic analysis
- :doc:`/components/noise_signals` - Available signal components
- :doc:`/advanced/batched_interface` - ArrayLikelihood details
