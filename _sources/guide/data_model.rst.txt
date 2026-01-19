Data Model
==========

Discovery's data model consists of two fundamental abstractions: **Kernel** objects and **GP** (Gaussian Process) objects.

Kernel Objects
--------------

Think of a ``Kernel`` as a noise matrix ``N``, which can be:

- **Inverted**: Apply ``N^{-1}`` to a vector
- **Applied**: Compute ``N^{-1} y`` for data vector ``y``
- **Sandwiched**: Evaluate the log-likelihood term ``y^T N^{-1} y``

Kernels represent noise components in the timing model, such as:

- White noise (EFAC, EQUAD)
- Measurement uncertainties
- Combined noise structures

Example Kernel Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import discovery as ds

   # Create measurement noise kernel
   noise = ds.makenoise_measurement(psr, noisedict)

   # The kernel can be applied in likelihood computations
   # log L ∝ -0.5 * y^T N^{-1} y

GP Objects
----------

A ``GP`` (Gaussian Process) object consists of:

- **Basis matrix** ``F`` (size ``ntoas × ngp``)
- **Prior/kernel** ``Phi`` (covariance in GP coefficient space)

GPs represent stochastic signals in the timing model, such as:

- Red noise (intrinsic timing noise)
- Dispersion measure variations
- Common processes across pulsars
- Gravitational wave backgrounds

Before Marginalization
~~~~~~~~~~~~~~~~~~~~~~

The GP introduces latent coefficients ``a`` that relate to the data through the basis:

.. math::

   y = F a + \epsilon

where :math:`\epsilon \sim \mathcal{N}(0, N)` is the noise. The joint distribution over data and coefficients is:

.. math::

   p(y, a | \Lambda) = p(y | a) \, p(a | \Lambda)

with the prior on coefficients:

.. math::

   p(a | \Lambda) = \mathcal{N}(a | 0, \Phi(\Lambda))

Here :math:`\Lambda` represents the hyperparameters (e.g., amplitude, spectral index) that
determine the covariance :math:`\Phi`. **In general, our free parameters live in** :math:`\Phi`
**through** :math:`\Lambda`.

Discovery provides access to these coefficients through conditional sampling (see :doc:`/advanced/conditional_sampling`).

After Marginalization
~~~~~~~~~~~~~~~~~~~~~

In general, we marginalize over the coefficients ``a`` to obtain the likelihood:

.. math::

   p(y | \Lambda) = \int p(y | a) \, p(a | \Lambda) \, da

This yields the marginalized log-likelihood:

.. math::

   \log p(y | \Lambda) = -\frac{1}{2} \left[ y^T C^{-1} y + \log |C| + n \log(2\pi) \right]

where :math:`C = N + F \Phi(\Lambda) F^T` is the marginalized covariance combining noise and
signal contributions. The log-determinant term :math:`\log |C|` contains the dependence on the
hyperparameters :math:`\Lambda` through :math:`\Phi(\Lambda)`.

For a more complete discussion of Gaussian process regression in pulsar timing,
see Chapter 7 of `Taylor (2021) <https://arxiv.org/abs/2105.13270>`_.

Example GP Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a red noise GP with Fourier basis
   rn_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')

   # Create a DM variation GP
   dm_gp = ds.makegp_fourier(psr, ds.powerlaw, components=30,
                             fourierbasis=ds.dmfourierbasis, name='dmgp')

WoodburyKernel
--------------

The ``WoodburyKernel`` combines a noise kernel and a GP:

.. code-block:: python

   combined = ds.WoodburyKernel(N, F, Phi)

This object efficiently represents the marginalized joint covariance:

.. math::

   C = N + F \Phi F^T

Using the `Woodbury matrix identity <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_,
the inverse can be computed efficiently:

.. math::

   (N + F \Phi F^T)^{-1} = N^{-1} - N^{-1} F (\Phi^{-1} + F^T N^{-1} F)^{-1} F^T N^{-1}

This avoids explicitly forming or inverting the full ``ntoas × ntoas`` covariance matrix,
which would be computationally expensive for large datasets.

Model Building
--------------

Discovery builds likelihoods by composing these components:

.. code-block:: python

   signals = [
       psr.residuals,                                    # Data vector y
       ds.makenoise_measurement(psr, noisedict),        # Kernel N
       ds.makegp_ecorr(psr, noisedict),                 # GP for ECORR
       ds.makegp_timing(psr),                            # GP for timing model
       ds.makegp_fourier(psr, ds.powerlaw, 30)          # GP for red noise
   ]

   likelihood = ds.PulsarLikelihood(signals)

The ``PulsarLikelihood`` object automatically constructs the appropriate
nested ``WoodburyKernel`` structure and provides a JAX-ready ``logL`` function.

See Also
--------

- :doc:`/tutorials/basic_likelihood` - Building likelihoods in practice
- :doc:`/components/noise_signals` - Available noise and signal components
- :doc:`/api/matrix` - Low-level kernel and GP implementations
- :doc:`/advanced/conditional_sampling` - Accessing GP coefficients
