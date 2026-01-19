Noise and Signal Components
============================

Discovery provides modular noise and signal components that can be combined to build likelihoods.

Measurement Noise
-----------------

White Noise (EFAC/EQUAD)
~~~~~~~~~~~~~~~~~~~~~~~~

Create measurement noise kernels with EFAC and EQUAD parameters:

.. code-block:: python

   # Backend-multiplexed (Enterprise style)
   noise = ds.makenoise_measurement(psr, noisedict)

   # Simple (no backend multiplexing)
   noise = ds.makenoise_measurement_simple(psr, noisedict)

   # With kernel ECORR (faster, less memory)
   noise = ds.makenoise_measurement(psr, noisedict, ecorr=True)

Parameters are multiplexed to pulsar and backend (e.g., ``B1855+09_430_ASP_efac``).

**Important:** If any single parameter needed for the model is not in ``noisedict``,
then **all** parameters become free (unfrozen). Otherwise, parameters in ``noisedict``
are frozen to their specified values.

See :func:`~discovery.signals.makenoise_measurement` and
:func:`~discovery.signals.makenoise_measurement_simple`.

ECORR
~~~~~

Epoch-averaged correlated noise can be added in two ways:

**Kernel ECORR (recommended):**

.. code-block:: python

   # Include ECORR in the noise kernel
   noise = ds.makenoise_measurement(psr, noisedict, ecorr=True)

Kernel ECORR is faster and uses less memory by incorporating ECORR directly
into the noise matrix.

**GP ECORR:**

.. code-block:: python

   # Backend-multiplexed
   ecorr_gp = ds.makegp_ecorr(psr, noisedict)

   # Simple (no backend multiplexing)
   ecorr_gp = ds.makegp_ecorr_simple(psr, noisedict)

GP ECORR can be useful when you need a diagonal noise matrix, for example in
outlier analysis (see `Wang and Taylor 2021 <https://arxiv.org/abs/2112.05698>`_).

ECORR uses quantization to define epochs and models correlated noise within epochs.

See :func:`~discovery.signals.makegp_ecorr` and
:func:`~discovery.signals.makegp_ecorr_simple`.

Stochastic Signals (GPs)
-------------------------

Fourier Basis GPs
~~~~~~~~~~~~~~~~~

Create GPs with Fourier basis (for red noise, DM variations, etc.):

.. code-block:: python

   # Red noise
   rn_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30,
                             name='rednoise')

   # DM variations
   dm_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30,
                             fourierbasis=ds.dmfourierbasis,
                             name='dmgp')

   # Common process (shared parameters)
   crn_gp = ds.makegp_fourier(psr, ds.powerlaw, components=14,
                              common=['crn_log10_A', 'crn_gamma'],
                              name='crn')

The Fourier basis has shape ``ntoas × 2*components`` (interleaved sines and cosines).

**Parameters:**

- ``prior``: JAX function with signature ``prior(f, df, arg1, ...)``, returning the power spectral density (PSD) at each frequency ``f``
- ``components``: Number of Fourier modes
- ``T``: Timespan (defaults to pulsar span)
- ``fourierbasis``: Basis function (defaults to :func:`~discovery.signals.fourierbasis`)
- ``common``: List of parameters shared across pulsars
- ``name``: Parameter name prefix

**Example prior function:**

.. code-block:: python

   def powerlaw(f, df, log10_A, gamma):
       """Power-law spectrum.

       Returns PSD at frequencies f.
       """
       return 10**(2*log10_A) * f**(-gamma) * (365.25*86400)**(-gamma+3) / (12*np.pi**2) * df

See :func:`~discovery.signals.makegp_fourier` and :doc:`priors_spectra` for available priors.

Global GPs
~~~~~~~~~~

Create correlated GPs for multiple pulsars:

.. code-block:: python

   # Hellings-Downs correlated GW background
   gw_gp = ds.makegp_fourier_global(psrs, ds.powerlaw, ds.hd_orf,
                                     components=14, T=Tspan,
                                     name='gw')

   # Monopole correlation
   mono_gp = ds.makegp_fourier_global(psrs, ds.powerlaw, ds.monopole_orf,
                                       components=14, T=Tspan,
                                       name='mono')

The ``orf`` parameter specifies the overlap reduction function (see :doc:`orf`).

For composite processes (e.g., HD + monopole):

.. code-block:: python

   gp = ds.makegp_fourier_global(
       psrs,
       [ds.powerlaw, ds.powerlaw],  # List of priors
       [ds.hd_orf, ds.monopole_orf], # List of ORFs
       components=14, T=Tspan,
       name='gw'
   )

Parameters are named ``{name}_{orf_name}_{argX}`` (e.g., ``gw_hd_orf_log10_A``).

See :func:`~discovery.signals.makegp_fourier_global`.

Common GPs (Batched)
~~~~~~~~~~~~~~~~~~~~

For ``ArrayLikelihood``, create vectorized GPs:

.. code-block:: python

   commongp = ds.makecommongp_fourier(psrs, ds.powerlaw,
                                      components=30, T=Tspan,
                                      name='red_noise')

This is similar to ``makegp_fourier`` but operates on a list of pulsars and
enables batched operations for GPU acceleration.

See :func:`~discovery.signals.makecommongp_fourier`.

Timing Model
~~~~~~~~~~~~

Add timing model GP:

.. code-block:: python

   # With SVD (recommended for stability)
   timing_gp = ds.makegp_timing(psr, svd=True)

   # With column normalization
   timing_gp = ds.makegp_timing(psr, svd=False)

   # With physical prior variance
   timing_gp = ds.makegp_timing(psr, svd=True, variance=1e-20)

When ``svd=True``, uses singular value decomposition for numerical stability.
When ``svd=False``, uses column normalization of the design matrix.
The ``variance`` parameter sets a proper Gaussian prior variance (in s²) instead
of an improper prior.

See :func:`~discovery.signals.makegp_timing`.

Fourier Basis Functions
------------------------

Standard Fourier Basis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   f, df, F = ds.fourierbasis(psr, components=30, T=None)

Returns:

- ``f``: Frequencies (Hz)
- ``df``: Frequency bin widths
- ``F``: Basis matrix (ntoas × 2*components)

The basis consists of interleaved sines and cosines at frequencies ``k/T``
for ``k = 1, ..., components``.

See :func:`~discovery.signals.fourierbasis`.

DM Fourier Basis
~~~~~~~~~~~~~~~~

For dispersion measure variations (chromatic index :math:`\\alpha = 2`):

.. code-block:: python

   f, df, F = ds.dmfourierbasis(psr, components=30, T=None, fref=1400)

Rescales the basis by ``(fref / psr.freqs)**2`` for DM scaling.

See :func:`~discovery.signals.dmfourierbasis`.

Chromatic Noise Basis
~~~~~~~~~~~~~~~~~~~~~

For general chromatic noise with arbitrary spectral index:

.. code-block:: python

   f, df, fmat_func = ds.dmfourierbasis_alpha(psr, components=30, T=None, fref=1400)

   # fmat_func is a closure that takes the chromatic index
   F = fmat_func(alpha)  # Returns basis scaled by (fref / psr.freqs)**alpha

The function returns a closure ``fmat_func(alpha)`` that generates the basis matrix
scaled by ``(fref / psr.freqs)**alpha`` for any chromatic index :math:`\\alpha`.
This allows the chromatic index to be a free parameter in the model.

**Example usage:**

.. code-block:: python

   # Create GP with variable chromatic index
   f, df, fmat_func = ds.dmfourierbasis_alpha(psr, components=30)

   def chromatic_prior(f, df, log10_A, gamma, alpha):
       # alpha is a free parameter here
       return ds.powerlaw(f, df, log10_A, gamma)

   # In the model, the basis will be evaluated at the sampled alpha value
   chrom_gp = ds.makegp_fourier(psr, chromatic_prior, components=30,
                                fourierbasis=lambda psr, c, T: (f, df, fmat_func),
                                name='chromatic')

See :func:`~discovery.signals.dmfourierbasis_alpha`.

Utility Functions
-----------------

Timespan
~~~~~~~~

Get the observing span:

.. code-block:: python

   # Single pulsar
   Tspan = ds.getspan(psr)

   # Multiple pulsars
   Tspan = ds.getspan(psrs)

See :func:`~discovery.signals.getspan`.

See Also
--------

- :doc:`priors_spectra` - GP prior functions
- :doc:`delays` - Deterministic delay models
- :doc:`/tutorials/basic_likelihood` - Building likelihoods
- :doc:`/api/signals` - Full API reference
