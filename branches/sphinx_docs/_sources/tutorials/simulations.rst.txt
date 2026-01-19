Simulations
===========

Discovery can generate synthetic pulsar timing data based on your noise and signal models.

Single Pulsar Simulation
-------------------------

Basic Setup
~~~~~~~~~~~

.. code-block:: python

   import discovery as ds
   import jax
   import numpy as np
   import os

   # Find data directory
   data_dir = os.path.join(ds.__path__[0], '..', '..', 'data')

   # Load a real pulsar to use its TOAs and observing setup
   psr = ds.Pulsar.read_feather(
       os.path.join(data_dir, 'v1p1_de440_pint_bipm2019-B1855+09.feather')
   )

   # Build likelihood with desired noise model
   psl = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
       ds.makegp_ecorr(psr, psr.noisedict),
       ds.makegp_timing(psr, svd=True, variance=1e-20),
       ds.makegp_fourier(psr, ds.powerlaw, 30, name='rednoise')
   ])

Generate Residuals
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get the sampler
   sampler = psl.sample

   # Sample parameters from prior
   params = ds.sample_uniform(sampler.params)

   # Or fix parameters to specific values
   params = {
       'B1855+09_rednoise_log10_A': -14.0,
       'B1855+09_rednoise_gamma': 4.33
   }

   # Generate residuals
   key = ds.rngkey(42)
   key, residuals = sampler(key, params)

   print(f"Generated {len(residuals)} residuals")

Update Pulsar Object
~~~~~~~~~~~~~~~~~~~~~

You can update the pulsar object with simulated data:

.. code-block:: python

   # Replace residuals (cast to numpy array)
   psr.residuals = np.array(residuals)

   # Create new likelihood with simulated data
   new_psl = ds.PulsarLikelihood([
       psr.residuals,
       ds.makenoise_measurement(psr, psr.noisedict),
       ds.makegp_ecorr(psr, psr.noisedict),
       ds.makegp_timing(psr, svd=True, variance=1e-20),
       ds.makegp_fourier(psr, ds.powerlaw, 30, name='rednoise')
   ])

   # Evaluate likelihood at true parameters
   logl = new_psl.logL
   logL_value = logl(params)

Pulsar Array Simulation
------------------------

For multi-pulsar simulations with correlated signals, use ``GlobalLikelihood``:

Setup Array Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import glob

   # Load pulsars
   data_pattern = os.path.join(data_dir, 'v1p1_de440_pint_bipm2019-*.feather')
   psrs = [ds.Pulsar.read_feather(f) for f in glob.glob(data_pattern)]

   Tspan = ds.getspan(psrs)

   # Build individual likelihoods
   pulsarlikelihoods = []
   for psr in psrs:
       psl = ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True, variance=1e-20),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ])
       pulsarlikelihoods.append(psl)

   # Add GW background
   gbl = ds.GlobalLikelihood(
       pulsarlikelihoods,
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

Generate Array Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get sampler
   sampler = gbl.sample

   # Sample parameters from prior
   params = ds.sample_uniform(sampler.params)

   # Optionally override specific parameters
   params['gw_gamma'] = 13/3
   params['gw_log10_A'] = -14.5

   # Generate residuals for all pulsars
   key = ds.rngkey(43)
   key, residuals = sampler(key, params)

   print(f"Generated residuals for {len(residuals)} pulsars")

The ``residuals`` returned by ``gbl.sample`` is a list of arrays, one for each pulsar.

Update Multiple Pulsars
~~~~~~~~~~~~~~~~~~~~~~~

To update individual pulsar objects:

.. code-block:: python

   # Update each pulsar (residuals is already a list of arrays)
   for psr, res in zip(psrs, residuals):
       psr.residuals = np.array(res)

   # Create new global likelihood with simulated data
   new_gbl = ds.GlobalLikelihood(
       [ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True, variance=1e-20),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ]) for psr in psrs],
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   # Evaluate at true parameters
   logl = new_gbl.logL
   print(f"Log-likelihood: {logl(params)}")

JIT-Compiling Samplers
-----------------------

For generating many realizations, JIT-compile the sampler:

.. code-block:: python

   sampler_jit = jax.jit(sampler)

   # First call compiles (slower)
   key, residuals = sampler_jit(key, params)

   # Subsequent calls are fast
   for i in range(100):
       key, residuals = sampler_jit(key, params)

Vectorized Simulations
~~~~~~~~~~~~~~~~~~~~~~

To generate multiple realizations in parallel:

.. code-block:: python

   import jax.numpy as jnp

   # Create a batch of parameters
   params_batch = jax.tree_map(
       lambda x: jnp.repeat(x[None, ...], 100, axis=0),
       params
   )

   # Create a batch of keys
   keys = jax.random.split(key, 100)

   # Vectorize sampler over batch
   sampler_vmap = jax.vmap(sampler, in_axes=(0, 0))

   # Generate 100 realizations at once
   keys, residuals_batch = sampler_vmap(keys, params_batch)

   print(f"Generated {len(residuals_batch)} realizations")
   print(f"Each realization has {len(residuals_batch[0])} pulsars")

Conditional Simulations
------------------------

You can also sample GP coefficients from their conditional distribution given data:

.. code-block:: python

   # Sample coefficients
   key = ds.rngkey(100)
   key, coeffs = gbl.sample_conditional(key, params)

   print("Sampled coefficients:")
   for name, coeff in coeffs.items():
       print(f"  {name}: shape {coeff.shape}")

See :doc:`/advanced/conditional_sampling` for more details on conditional sampling.

Complete Example
----------------

Here's a complete simulation workflow:

.. code-block:: python

   import discovery as ds
   import jax
   import numpy as np
   import glob
   import os

   # Find data
   data_dir = os.path.join(ds.__path__[0], '..', '..', 'data')
   data_pattern = os.path.join(data_dir, 'v1p1_de440_pint_bipm2019-*.feather')

   # Load pulsars
   psrs = [ds.Pulsar.read_feather(f) for f in glob.glob(data_pattern)[:5]]
   Tspan = ds.getspan(psrs)

   # Build model
   gbl = ds.GlobalLikelihood(
       [ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True, variance=1e-20),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ]) for psr in psrs],
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   # Sample parameters from prior
   params = ds.sample_uniform(gbl.sample.params)

   # Override GW parameters
   params['gw_gamma'] = 13/3
   params['gw_log10_A'] = -14.5

   # Generate data
   key = ds.rngkey(42)
   key, residuals = gbl.sample(key, params)

   # Update pulsars
   for psr, res in zip(psrs, residuals):
       psr.residuals = np.array(res)

   # Rebuild and evaluate
   new_gbl = ds.GlobalLikelihood(
       [ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True, variance=1e-20),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise')
       ]) for psr in psrs],
       globalgp=ds.makegp_fourier_global(
           psrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan,
           name='gw'
       )
   )

   logl = jax.jit(new_gbl.logL)
   print(f"Log-likelihood at true params: {logl(params)}")

See Also
--------

- :doc:`basic_likelihood` - Building likelihoods
- :doc:`/advanced/conditional_sampling` - GP coefficient sampling
- :doc:`/api/likelihood` - Likelihood API reference
