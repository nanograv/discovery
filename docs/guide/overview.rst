Overview
========

What is Discovery?
------------------

*Discovery* is a next-generation pulsar-timing-array data-analysis package, **built for speed** on a `JAX <https://jax.readthedocs.io/en/latest/>`_ backend that supports GPU execution and autodifferentiation.

Philosophy
----------

If `Enterprise <https://github.com/nanograv/enterprise>`_ is Spock—logical and elegant—*Discovery* is all Scotty: fast, efficient, and not above a hack if it gets you to warp speed.

Key Features
------------

**JAX Backend**
   Discovery is built on JAX, enabling:

   - Just-in-time (JIT) compilation for optimized execution
   - Automatic differentiation for gradient-based inference
   - Seamless GPU acceleration with CUDA
   - Efficient vectorization with vmap

**High Performance**
   - Optimized for modern hardware (GPUs, TPUs)
   - Batched operations for array-level analysis
   - Minimal overhead for large datasets

**Flexible Modeling**
   - Modular signal and noise components
   - Support for custom priors and spectra
   - Deterministic delays and stochastic processes
   - Global and pulsar-specific models

Requirements
------------

Discovery needs a modern Python environment with:

- ``numpy`` - Numerical computing
- ``scipy`` - Scientific computing
- ``jax`` - Autodiff and GPU support
- ``pyarrow`` - Efficient data storage

Discovery will be happier running on an Nvidia GPU with CUDA-enabled JAX.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Discovery's subpackages require additional dependencies:

- ``discovery.flow`` - Normalizing flow samplers
- ``discovery.samplers`` - Various sampling backends (numpyro, etc.)

See :doc:`/installation` for installation instructions.

Next Steps
----------

- :doc:`/guide/data_model` - Understand Discovery's core abstractions
- :doc:`/guide/pulsar_data` - Learn about data handling
- :doc:`/tutorials/basic_likelihood` - Build your first likelihood
