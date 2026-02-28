.. Discovery documentation master file

Welcome to Discovery's documentation!
======================================

Discovery is a next-generation PTA data analysis package built on JAX.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   tutorials/curn_example
   guide/overview
   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   quickstart
   tutorials/basic_likelihood
   tutorials/optimal_statistic
   tutorials/simulations

.. toctree::
   :maxdepth: 1
   :caption: Component Reference

   components/noise_signals
   components/priors_spectra
   components/delays

.. toctree::
   :maxdepth: 1
   :caption: Other Useful Information

   guide/data_model
   guide/pulsar_data

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   advanced/conditional_sampling

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples

Overview
========

Discovery provides advanced tools for pulsar timing array analysis, including:

- JAX-based likelihood computations for efficient gradient-based inference
- Flexible signal modeling with support for stochastic and deterministic signals
- Solar wind and chromatic delay models
- Integration with modern sampling frameworks
- GPU acceleration support

Getting Started
===============

New users should start with the :doc:`guide/overview` to understand Discovery's philosophy,
then proceed to :doc:`quickstart` for hands-on examples.

For detailed API documentation, see :doc:`api/index`.

Quick Links
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
