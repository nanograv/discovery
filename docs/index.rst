.. Discovery documentation master file

Welcome to Discovery's documentation!
======================================

Discovery is a next-generation pulsar-timing-array data analysis package built on JAX.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/overview
   guide/data_model
   guide/pulsar_data
   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   quickstart
   tutorials/basic_likelihood
   tutorials/simulations
   tutorials/optimal_statistic

.. toctree::
   :maxdepth: 2
   :caption: Component Reference

   components/noise_signals
   components/priors_spectra
   components/delays

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/conditional_sampling

.. toctree::
   :maxdepth: 2
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
