Deterministic Delays
====================

Discovery provides deterministic delay functions for modeling known physical effects that modify
the observed timing residuals. These delays are added directly to the likelihood model and can
have free or fixed parameters.

Creating Delay Functions
-------------------------

The ``makedelay`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a deterministic delay component:

.. code-block:: python

   delay = ds.makedelay(psr, delayfunc, common=None, name='delay')

This wraps a JAX-compatible delay function ``delayfunc`` to create a component that can be
included in a likelihood.

**Parameters:**

- ``psr``: Pulsar object
- ``delayfunc``: JAX function that computes the delay
- ``common``: List of parameter names shared across pulsars
- ``name``: Parameter name prefix

**Delay Function Signature:**

The delay function must have signature:

.. code-block:: python

   def delayfunc(arg1, arg2, ...):
       """Compute delay in seconds.

       Returns
       -------
       delay : array
           Timing delay at each TOA (seconds)
       """
       return ...

**Automatic Parameter Passing:**

If the first arguments of ``delayfunc`` are defined attributes of ``psr`` (like ``toas`` or ``freqs``),
they are automatically passed and excluded from the parameter list. These must come before all
variable parameters.

**Parameter Naming:**

Parameters are named ``{psrname}_{name}_{argX}`` unless included in the ``common`` list.

See :func:`~discovery.signals.makedelay`.

Solar Wind DM Delay
-------------------

1/r² Solar Wind Model
~~~~~~~~~~~~~~~~~~~~~

Model dispersion measure variations due to solar wind:

.. code-block:: python

   solardm_func = ds.make_solardm(psr)
   delay = ds.makedelay(psr, solardm_func, name='solardm')

See :func:`~discovery.solar.make_solardm` and :func:`~discovery.signals.makedelay`.

Implements a :math:`1/r^2` solar wind density model. The delay function has signature:

.. code-block:: python

   solardm_func(n_earth)

where ``n_earth`` is the electron density at Earth's orbit (cm⁻³).

**Physical Model:**

The solar wind introduces a frequency-dependent delay:

.. math::

   \Delta t(f) = \frac{n_{\mathrm{Earth}}}{f^2} \cdot \frac{\mathrm{AU}}{R_\oplus \sin(\theta_{\mathrm{impact}})} \cdot 4.148808 \times 10^3

where:

- :math:`n_{\mathrm{Earth}}` is the electron density at 1 AU (free parameter)
- :math:`R_\oplus` is Earth's distance from the Sun
- :math:`\theta_{\mathrm{impact}}` is the solar impact angle
- :math:`f` is the observing frequency (MHz)

The function pre-computes the geometric factors using the pulsar's position and Earth's ephemeris.

**Parameter:**

- ``{psrname}_solardm_n_earth``: Electron density at 1 AU (cm⁻³)

Typical values: :math:`n_{\mathrm{Earth}} \sim 5` cm⁻³.

Solar Wind GP
~~~~~~~~~~~~~

For stochastic fluctuations on top of the deterministic model:

.. code-block:: python

   # Add deterministic mean
   solardm_func = ds.make_solardm(psr)
   delay = ds.makedelay(psr, solardm_func, name='solardm')

   # Add stochastic fluctuations
   f, df, fmat_sw = ds.make_solardmfourierbasis(psr, components=30)
   sw_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30,
                              fourierbasis=lambda p, c, T: (f, df, fmat_sw),
                              name='sw_gp')

See :func:`~discovery.solar.make_solardmfourierbasis`, :func:`~discovery.signals.makegp_fourier`,
and :func:`~discovery.signals.powerlaw`.

The ``make_solardmfourierbasis`` function creates a Fourier basis scaled by the solar wind
geometry, so the GP represents fluctuations in :math:`n_{\mathrm{Earth}}` around the mean.

Chromatic Delays
----------------

Exponential Dip Model
~~~~~~~~~~~~~~~~~~~~~

Model chromatic exponential dips - sudden radio frequency-dependent advances of pulse arrival times.
These events can impact measurements of time-correlated signals (see `Hazboun et al. 2020 <https://arxiv.org/abs/1909.08644>`_).

.. code-block:: python

   decay_func = ds.make_chromaticdelay(psr)
   delay = ds.makedelay(psr, decay_func, name='chromatic')

See :func:`~discovery.solar.make_chromaticdelay` and :func:`~discovery.signals.makedelay`.

The delay function has signature:

.. code-block:: python

   decay_func(t0, log10_Amp, log10_tau, idx)

**Physical Model:**

Implements an exponential decay with chromatic scaling:

.. math::

   \Delta t(t, f) = \begin{cases}
   -10^{\log_{10}(A)} \cdot e^{-(t - t_0) / \tau} \cdot \left(\frac{1400}{f}\right)^\alpha & t > t_0 \\
   0 & t \leq t_0
   \end{cases}

where:

- :math:`t_0` is the event time (MJD)
- :math:`A = 10^{\log_{10}(A)}` is the amplitude (seconds at 1400 MHz)
- :math:`\tau = 10^{\log_{10}(\tau)}` is the decay timescale (days)
- :math:`\alpha` is the chromatic index (2 for DM, 4 for scattering)
- :math:`f` is the observing frequency (MHz)

**Parameters:**

- ``{psrname}_chromatic_t0``: Event epoch (MJD)
- ``{psrname}_chromatic_log10_Amp``: Log amplitude
- ``{psrname}_chromatic_log10_tau``: Log decay timescale (days)
- ``{psrname}_chromatic_idx``: Chromatic index

**Fixed Chromatic Index:**

To fix the chromatic index:

.. code-block:: python

   decay_func = ds.make_chromaticdelay(psr, idx=2.0)  # Fix to DM scaling

This removes ``idx`` from the parameter list. See also :func:`~discovery.solar.chromaticdelay`
for the underlying delay function.

Binary Black Hole Signals
--------------------------

Continuous Wave from BBH
~~~~~~~~~~~~~~~~~~~~~~~~

Model deterministic signals from binary black hole systems:

.. code-block:: python

   # With pulsar term
   bbh_func = ds.makedelay_binary(pulsarterm=True)
   delay = ds.makedelay(psr, bbh_func, name='bbh')

   # Without pulsar term (Earth term only)
   bbh_func = ds.makedelay_binary(pulsarterm=False)
   delay = ds.makedelay(psr, bbh_func, name='bbh')

See :func:`~discovery.deterministic.makedelay_binary` and :func:`~discovery.signals.makedelay`.

**Parameters:**

- ``log10_h0``: Log strain amplitude
- ``log10_f0``: Log GW frequency (Hz)
- ``ra``: Right ascension (rad)
- ``sindec``: Sine of declination
- ``cosinc``: Cosine of inclination angle
- ``psi``: Polarization angle (rad)
- ``phi_earth``: Initial phase at Earth (rad)
- ``phi_psr``: Initial phase at pulsar (rad, only if ``pulsarterm=True``)

The signal is computed using the Earth term (and optionally pulsar term) for a circular binary
emitting gravitational waves. Based on the formalism from Ellis et al. (2012, 2013).

Fourier-Domain BBH
~~~~~~~~~~~~~~~~~~

For signals in Fourier space (useful with FFTCov):

.. code-block:: python

   bbh_fourier = ds.makefourier_binary(pulsarterm=True)

See :func:`~discovery.deterministic.makefourier_binary`.

This computes the Fourier components directly, which can be more efficient for certain
likelihood computations.

Custom Delay Functions
----------------------

You can define custom JAX-compatible delay functions:

.. code-block:: python

   import jax.numpy as jnp

   def quadratic_spindown(toas, f0, f1, pepoch):
       """Model timing noise as quadratic spindown.

       Parameters
       ----------
       toas : array
           Times of arrival (seconds)
       f0 : float
           Spin frequency (Hz)
       f1 : float
           Spin frequency derivative (Hz/s)
       pepoch : float
           Reference epoch (MJD)
       """
       dt = (toas / 86400.0) - pepoch  # Convert to days
       phase = 2.0 * jnp.pi * (f0 * dt + 0.5 * f1 * dt**2)
       return phase / (2.0 * jnp.pi * f0)  # Convert phase to time

   # Use in model
   delay = ds.makedelay(psr, quadratic_spindown, name='spindown')

See :func:`~discovery.signals.makedelay`.

**Requirements:**

- Must be JAX-compatible (use ``jax.numpy`` instead of ``numpy``)
- Must return delay in seconds with same shape as ``toas``
- Pulsar attributes (``toas``, ``freqs``, ``pos``, etc.) can be first arguments
- All operations must be differentiable for gradient-based inference

Examples
--------

Single Pulsar with Solar Wind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import discovery as ds

   # Load pulsar
   psr = ds.Pulsar.read_feather('data/v1p1_de440_pint_bipm2019-B1855+09.feather')

   # Solar wind delay
   solardm_func = ds.make_solardm(psr)
   delay = ds.makedelay(psr, solardm_func, name='solardm')

   # Build likelihood
   signals = [
       psr.residuals,
       ds.makenoise_measurement(psr, noisedict),
       ds.makegp_timing(psr, svd=True),
       ds.makegp_fourier(psr, ds.powerlaw, 30, name='rn'),
       delay
   ]

   logl = ds.PulsarLikelihood(signals)

   # Parameters: rn_log10_A, rn_gamma, B1855+09_solardm_n_earth

Multiple Chromatic Events
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Model two separate DM events
   decay1 = ds.make_chromaticdelay(psr, idx=2.0)
   delay1 = ds.makedelay(psr, decay1, name='dm_event1')

   decay2 = ds.make_chromaticdelay(psr, idx=2.0)
   delay2 = ds.makedelay(psr, decay2, name='dm_event2')

   signals = [
       psr.residuals,
       ds.makenoise_measurement(psr, noisedict),
       ds.makegp_timing(psr, svd=True),
       delay1,
       delay2
   ]

   logl = ds.PulsarLikelihood(signals)

   # Parameters include: dm_event1_t0, dm_event1_log10_Amp, dm_event1_log10_tau
   #                     dm_event2_t0, dm_event2_log10_Amp, dm_event2_log10_tau

Common Solar Wind (PTA)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Shared n_earth across all pulsars
   delays = []
   for psr in psrs:
       solardm_func = ds.make_solardm(psr)
       delay = ds.makedelay(psr, solardm_func,
                           common=['n_earth'], name='solardm')
       delays.append(delay)

   # Build PTA likelihood
   gbl = ds.GlobalLikelihood(psrs, noisedict, delays=delays)

   # Single parameter: n_earth (shared across all pulsars)

See Also
--------

- :doc:`noise_signals` - Stochastic signal components
- :doc:`priors_spectra` - GP prior functions
- :doc:`/tutorials/basic_likelihood` - Building likelihoods
- :doc:`/api/signals` - Signals API reference
- :doc:`/api/solar` - Solar wind functions
- :doc:`/api/deterministic` - Deterministic signals API
