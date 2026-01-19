Optimal Statistic
=================

The optimal statistic (OS) provides a rapid first-pass detection method for gravitational wave backgrounds in pulsar timing arrays.

Creating an OS Object
---------------------

The OS requires a ``GlobalLikelihood`` where each pulsar has a GP component named ``'gw'`` with at least the common parameter ``gw_log10_A``:

.. code-block:: python

   import discovery as ds
   import glob
   import os

   # Find and load data
   data_dir = os.path.join(ds.__path__[0], '..', '..', 'data')
   data_pattern = os.path.join(data_dir, 'v1p1_de440_pint_bipm2019-*.feather')
   psrs = [ds.Pulsar.read_feather(f) for f in glob.glob(data_pattern)]

   Tspan = ds.getspan(psrs)

   # Build likelihood with 'gw' component
   gbl = ds.GlobalLikelihood(
       [ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan,
                            name='rednoise'),
           ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                            common=['gw_log10_A', 'gw_gamma'],
                            name='gw')
       ]) for psr in psrs]
   )

   # Create OS object
   os_obj = ds.OS(gbl)

Note: The ``globalgp`` in the ``GlobalLikelihood`` is unused by the OS—it uses the
per-pulsar ``'gw'`` components.

Computing the Optimal Statistic
--------------------------------

Basic Computation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set parameters (must include all likelihood parameters)
   params = ds.sample_uniform(gbl.logL.params)

   # Compute OS
   result = os_obj.os(params)

   print(f"OS: {result['os']}")
   print(f"OS sigma: {result['os_sigma']}")
   print(f"SNR: {result['snr']}")
   print(f"log10_A: {result['log10_A']}")

The result dictionary contains:

- ``'os'``: Optimal statistic value
- ``'os_sigma'``: Standard deviation
- ``'snr'``: Signal-to-noise ratio (OS / OS sigma)
- ``'log10_A'``: Reconstructed GW amplitude

Overlap Reduction Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the OS uses Hellings-Downs correlation (``hd_orfa``). You can specify others:

.. code-block:: python

   # Monopole correlation
   result_mono = os_obj.os(params, orfa=ds.monopole_orfa)

   # Dipole correlation
   result_dip = os_obj.os(params, orfa=ds.dipole_orfa)

   # Hellings-Downs (default)
   result_hd = os_obj.os(params, orfa=ds.hd_orfa)

JIT Compilation
---------------

The OS can be JIT-compiled for performance:

.. code-block:: python

   import jax

   # JIT-compile
   os_jit = jax.jit(os_obj.os)

   # For using non-default orfa, specify static argument
   os_mono_jit = jax.jit(os_obj.os, static_argnums=1)
   result = os_mono_jit(params, ds.monopole_orfa)

Vectorization
~~~~~~~~~~~~~

Compute OS for many parameter samples in parallel:

.. code-block:: python

   import jax.numpy as jnp

   # Create batch of parameters (each key has array of values)
   nsamples = 100
   params_batch = {}
   for key in gbl.logL.params:
       param_samples = jnp.array([ds.sample_uniform([key])[key]
                                   for _ in range(nsamples)])
       params_batch[key] = param_samples

   # Vectorize over parameters (axis 0 of each dict value)
   os_vmap = jax.vmap(os_obj.os, in_axes=(0, None))

   # Compute for all samples
   results = os_vmap(params_batch, ds.hd_orfa)

   print(f"SNRs: {results['snr']}")
   print(f"Mean SNR: {results['snr'].mean()}")

Scrambling Analysis
-------------------

Test significance by scrambling pulsar positions:

.. code-block:: python

   import numpy as np

   # Original result
   result_true = os_obj.os(params)

   # Create scrambled positions (random on sphere)
   npsr = len(psrs)
   phi = np.random.uniform(0, 2*np.pi, npsr)
   theta = np.arccos(np.random.uniform(-1, 1, npsr))

   positions = np.array([
       np.sin(theta) * np.cos(phi),
       np.sin(theta) * np.sin(phi),
       np.cos(theta)
   ]).T

   # Compute OS with scrambled positions
   result_scrambled = os_obj.scramble(params, positions)

   print(f"True SNR: {result_true['snr']}")
   print(f"Scrambled SNR: {result_scrambled['snr']}")

Vectorized Scrambling
~~~~~~~~~~~~~~~~~~~~~

Generate many scrambled realizations:

.. code-block:: python

   # Generate 1000 scrambled position sets
   nscrambles = 1000
   phi = np.random.uniform(0, 2*np.pi, (nscrambles, npsr))
   theta = np.arccos(np.random.uniform(-1, 1, (nscrambles, npsr)))

   positions_batch = np.array([
       np.sin(theta) * np.cos(phi),
       np.sin(theta) * np.sin(phi),
       np.cos(theta)
   ]).transpose(1, 2, 0)

   # Vectorize over positions (axis 0)
   os_scramble_vmap = jax.vmap(os_obj.scramble, in_axes=(None, 0, None))

   # Compute all scrambles
   results_scrambled = os_scramble_vmap(params, positions_batch, ds.hd_orfa)

   # Compare to true value
   snr_true = result_true['snr']
   snr_scrambled = results_scrambled['snr']
   p_value = (snr_scrambled > snr_true).mean()

   print(f"True SNR: {snr_true}")
   print(f"p-value: {p_value}")

Phase Shifting
--------------

Test significance by shifting GW basis phases:

.. code-block:: python

   # Random phases for each pulsar and frequency (npsr × ngw)
   npsr = len(psrs)
   ngw = 14  # Number of GW frequencies
   phases = np.random.uniform(0, 2*np.pi, (npsr, ngw))

   # Compute OS with shifted phases
   result_shifted = os_obj.shift(params, phases)

   print(f"Shifted SNR: {result_shifted['snr']}")

Vectorized Phase Shifting
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate many phase realizations
   nshifts = 1000
   phases_batch = np.random.uniform(0, 2*np.pi, (nshifts, npsr, ngw))

   # Vectorize over phases (axis 0)
   os_shift_vmap = jax.vmap(os_obj.shift, in_axes=(None, 0, None))

   # Compute all shifts
   results_shifted = os_shift_vmap(params, phases_batch, ds.hd_orfa)

   # Compute p-value
   p_value = (results_shifted['snr'] > result_true['snr']).mean()

CDF Computation
---------------

Compute the cumulative distribution function using the generalized chi-squared (GX2) distribution:

.. code-block:: python

   # SNR values to evaluate CDF at
   xs = np.linspace(0, 5, 100)

   # Compute CDF
   cdf_values = os_obj.gx2cdf(params, xs, cutoff=1e-6, limit=100, epsabs=1e-6)

   # Plot (requires matplotlib)
   import matplotlib.pyplot as plt

   plt.plot(xs, cdf_values)
   plt.xlabel('SNR')
   plt.ylabel('CDF')
   plt.title('OS SNR Distribution')
   plt.show()

Parameters:

- ``cutoff``: If float, exclude eigenvalues smaller than this; if int, keep only the largest ``cutoff`` eigenvalues
- ``limit``, ``epsabs``: Passed to ``scipy.integrate.quad``

Note: ``gx2cdf`` currently cannot be JIT-compiled or vmapped.

Complete Example
----------------

Here's a complete OS analysis with significance testing:

.. code-block:: python

   import discovery as ds
   import jax
   import numpy as np
   import glob
   import os

   # Load data
   data_dir = os.path.join(ds.__path__[0], '..', '..', 'data')
   data_pattern = os.path.join(data_dir, 'v1p1_de440_pint_bipm2019-*.feather')
   psrs = [ds.Pulsar.read_feather(f) for f in glob.glob(data_pattern)]
   Tspan = ds.getspan(psrs)

   # Build likelihood
   gbl = ds.GlobalLikelihood([
       ds.PulsarLikelihood([
           psr.residuals,
           ds.makenoise_measurement(psr, psr.noisedict),
           ds.makegp_ecorr(psr, psr.noisedict),
           ds.makegp_timing(psr, svd=True),
           ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='rednoise'),
           ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                            common=['gw_log10_A', 'gw_gamma'], name='gw')
       ]) for psr in psrs
   ])

   # Create OS
   os_obj = ds.OS(gbl)

   # Sample parameters
   params = ds.sample_uniform(gbl.logL.params)

   # Compute OS
   result = os_obj.os(params)
   print(f"OS: {result['os']:.3f}")
   print(f"SNR: {result['snr']:.3f}")
   print(f"log10_A: {result['log10_A']:.3f}")

   # Scrambling test
   npsr, ngw = len(psrs), 14
   nscrambles = 1000

   phi = np.random.uniform(0, 2*np.pi, (nscrambles, npsr))
   theta = np.arccos(np.random.uniform(-1, 1, (nscrambles, npsr)))
   positions = np.array([
       np.sin(theta) * np.cos(phi),
       np.sin(theta) * np.sin(phi),
       np.cos(theta)
   ]).transpose(1, 2, 0)

   os_scramble = jax.vmap(os_obj.scramble, in_axes=(None, 0, None))
   results_scrambled = os_scramble(params, positions, ds.hd_orfa)

   p_value = (results_scrambled['snr'] > result['snr']).mean()
   print(f"p-value (scrambling): {p_value:.4f}")

See Also
--------

- :doc:`basic_likelihood` - Building likelihoods
- :doc:`/components/orf` - Overlap reduction functions
- :doc:`/api/optimal` - Optimal statistic API reference
