GP Priors and Spectra
======================

Discovery provides prior functions for Gaussian processes that define power spectral densities (PSDs) or covariance functions.

Fourier-Based Prior Function Interface
---------------------------------------

For Fourier-basis GPs (see :doc:`/guide/data_model`), prior functions define the power spectral density,
which produces the diagonal of the covariance matrix :math:`\Phi`:

.. code-block:: python

   def prior(f, df, *args):
       """
       Parameters
       ----------
       f : array
           Frequencies (Hz)
       df : array
           Frequency bin widths
       *args : scalars
           Hyperparameters (e.g., log10_A, gamma)

       Returns
       -------
       psd : array
           Power spectral density at each frequency
       """
       return ...

The PSD represents the variance at each Fourier frequency. For a Fourier basis with frequencies
:math:`f_k`, the covariance matrix is approximated as:

.. math::

   \Phi_{kk'} \approx \delta_{kk'} S(f_k) \Delta f_k

where :math:`S(f_k)` is the PSD and :math:`\delta_{kk'}` is the Kronecker delta (diagonal matrix).
This assumes Fourier modes are independent, which is an approximation for finite-length data.

See :doc:`/guide/data_model` for the mathematical framework.

Standard Priors
---------------

Power Law
~~~~~~~~~

The standard power-law spectrum for red noise and gravitational waves:

.. code-block:: python

   psd = ds.powerlaw(f, df, log10_A, gamma)

Implements:

.. math::

   S(f) = \\frac{10^{2 \log_{10}(A)}}{12\pi^2} f^{-\gamma} T_{\mathrm{yr}}^{\gamma-3} \, \Delta f

where :math:`\log_{10}(A)` = ``log10_A`` is the log-amplitude, :math:`\gamma` = ``gamma`` is the
spectral index, and :math:`T_{\mathrm{yr}} = 365.25 \times 86400` s converts to per-year normalization.

See :func:`~discovery.signals.powerlaw`.

Free Spectrum
~~~~~~~~~~~~~

Independent amplitudes at each frequency:

.. code-block:: python

   psd = ds.freespectrum(f, df, log10_rho)

where ``log10_rho`` is a vector (length = ``components``) of log-amplitudes.

Implements:

.. math::

   S(f_k) = 10^{2 \log_{10}(\rho_k)}

The function signature uses a type annotation to indicate ``log10_rho`` should be
treated as a vector parameter:

.. code-block:: python

   def freespectrum(f, df, log10_rho: typing.Sequence):
       return 10**(2*log10_rho)

The ``typing.Sequence`` annotation tells Discovery to treat this parameter as a vector
in the parameter dictionary (otherwise it assumes scalar parameters). Discovery automatically
creates parameter names like ``B1855+09_freespectrum_log10_rho(30)`` when ``components=30``.

See :func:`~discovery.signals.freespectrum`.

Combined Priors
---------------

Power Law + Common Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models with intrinsic red noise plus a common process:

.. code-block:: python

   prior = ds.makepowerlaw_crn(components_crn=14)

   # Use in GP
   gp = ds.makegp_fourier(psr, prior, components=30,
                          common=['crn_log10_A', 'crn_gamma'],
                          name='rednoise')

This creates a prior function with signature:

.. code-block:: python

   prior(f, df, log10_A, gamma, crn_log10_A, crn_gamma)

The first ``components_crn`` frequencies use the combined spectrum:

.. math::

   S(f) = S_{\mathrm{RN}}(f; A, \gamma) + S_{\mathrm{CRN}}(f; A_{\mathrm{crn}}, \gamma_{\mathrm{crn}})

and the remaining frequencies use only the red noise spectrum.

**Parameter names:**

- Intrinsic: ``{psrname}_rednoise_log10_A``, ``{psrname}_rednoise_gamma``
- Common: ``crn_log10_A``, ``crn_gamma``

This is particularly useful with :class:`~discovery.likelihood.ArrayLikelihood` as it
enables vectorization across pulsars while maintaining both intrinsic and common processes.

See :func:`~discovery.signals.makepowerlaw_crn`.

Time-Domain Covariance (FFTCov)
--------------------------------

In reality, for finite-length data, Fourier frequencies are not truly independent, and the
covariance matrix :math:`\Phi` should not be diagonal. The FFTCov approach provides an efficient
way to estimate the dense (non-diagonal) covariance matrix from the power spectral density.

.. code-block:: python

   # Single pulsar
   gp = ds.makegp_fftcov(psr, ds.powerlaw, components=30, T=Tspan,
                         t0=tmin, order=1, name='rednoise')

   # Common GP (batched)
   commongp = ds.makecommongp_fftcov(psrs, ds.powerlaw, components=30, T=Tspan,
                                      t0=tmin, order=1, name='rednoise')

   # Global GP with correlation
   globalgp = ds.makeglobalgp_fftcov(psrs, ds.powerlaw, ds.hd_orf,
                                      components=30, T=Tspan, t0=tmin,
                                      order=1, name='gw')

**Parameters:**

- ``prior``: PSD function (same interface as Fourier-basis)
- ``components``: Number of time-interpolation modes
- ``T``: Timespan
- ``t0``: Reference time (start of observations)
- ``order``: Interpolation order (0=nearest, 1=linear)
- ``oversample``: FFT oversampling factor (default=3)
- ``fmax_factor``: Maximum frequency factor (default=1)
- ``cutoff``: Eigenvalue cutoff for dimensionality reduction

The FFTCov approach computes the time-domain covariance function from the PSD via inverse FFT,
then uses time-interpolated basis functions to construct the GP efficiently.

For details, see `Crisostomi et al. (2025) <https://arxiv.org/abs/2506.13866>`_.

See :func:`~discovery.signals.makegp_fftcov`,
:func:`~discovery.signals.makecommongp_fftcov`, and
:func:`~discovery.signals.makeglobalgp_fftcov`.

Custom Priors
-------------

You can define custom JAX-compatible prior functions:

.. code-block:: python

   import jax.numpy as jnp

   def broken_powerlaw(f, df, log10_A, gamma_low, gamma_high, log10_fb):
       """Broken power law at frequency fb."""
       fb = 10**log10_fb
       S_low = 10**(2*log10_A) * f**(-gamma_low)
       S_high = 10**(2*log10_A) * fb**(gamma_high - gamma_low) * f**(-gamma_high)

       # Smooth transition
       return jnp.where(f < fb, S_low, S_high) * (365.25*86400)**(-gamma_low+3) / (12*jnp.pi**2) * df

   # Use in model
   gp = ds.makegp_fourier(psr, broken_powerlaw, components=30,
                          name='broken_rn')

**Requirements:**

- Must be JAX-compatible (use ``jax.numpy`` instead of ``numpy``)
- First two arguments must be ``f`` and ``df``
- Return PSD array with same shape as ``f``
- All operations must be differentiable for gradient-based inference

Parameter Priors
----------------

For uniform priors on hyperparameters:

.. code-block:: python

   # Default priors (includes standard parameter names)
   logprior = ds.makelogprior_uniform(logl.params)

   # Custom priors
   priordict = {
       'B1855+09_rednoise_log10_A': [-18, -11],
       'B1855+09_rednoise_gamma': [0, 7],
       'crn_log10_A': [-18, -11],
       'crn_gamma': [0, 7]
   }
   logprior = ds.makelogprior_uniform(logl.params, priordict)

   # Sample from prior
   params = ds.sample_uniform(logl.params, priordict)

Discovery includes a standard prior dictionary :data:`~discovery.prior.priordict_standard`
with default uniform (or log-uniform) prior ranges for common parameter names:

- ``*_log10_A``: [-18, -11]
- ``*_gamma``: [0, 7]
- ``gw_log10_A``, ``crn_log10_A``: [-18, -11]
- ``gw_gamma``, ``crn_gamma``: [0, 7]

You can access this dictionary directly:

.. code-block:: python

   print(ds.priordict_standard)

See :func:`~discovery.prior.makelogprior_uniform`,
:func:`~discovery.prior.sample_uniform`, and
:data:`~discovery.prior.priordict_standard`.

Examples
--------

Standard Red Noise + CRN
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import discovery as ds

   # Power law + common process
   prior = ds.makepowerlaw_crn(14)

   gp = ds.makegp_fourier(psr, prior, components=30,
                          common=['crn_log10_A', 'crn_gamma'],
                          name='rednoise')

   # Parameters: rednoise_log10_A, rednoise_gamma, crn_log10_A, crn_gamma

DM Variations
~~~~~~~~~~~~~

.. code-block:: python

   # DM with power law
   dm_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30,
                             fourierbasis=ds.dmfourierbasis,
                             name='dm')

   # Parameters: dm_log10_A, dm_gamma

Free Spectrum
~~~~~~~~~~~~~

.. code-block:: python

   # Independent amplitudes at each frequency
   fs_gp = ds.makegp_fourier(psr, ds.freespectrum, components=30,
                             name='fs')

   # Parameters: fs_log10_rho(30) (vector parameter)

See Also
--------

- :doc:`noise_signals` - Signal component functions
- :doc:`/guide/data_model` - GP mathematical framework
- :doc:`/tutorials/basic_likelihood` - Using priors in likelihoods
- :doc:`/api/signals` - Full API reference
- :doc:`/api/prior` - Prior API reference
