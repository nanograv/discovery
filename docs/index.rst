.. Discovery documentation master file

Welcome to Discovery's documentation!
======================================

Discovery is a next-generation PTA data analysis package built on JAX.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   tutorials/curn_example
   guide/overview
   guide/model_summary
   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   quickstart
   tutorials/basic_likelihood
   tutorials/params_example
   tutorials/optimal_statistic
   tutorials/simulations
   tutorials/cookbook_models
   tutorials/cw_extsignal_example

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
   advanced/single_precision

.. toctree::
   :maxdepth: 1
   :caption: Development

   metamatrix
   metamatrix_dev
   design/README
   design/metamatrix/architecture
   design/metamatrix/deletion_checklist
   design/metamatrix/parity_coverage
   design/single_precision/README
   design/single_precision/future
   design/single_precision/research_refdelta
   design/single_precision/research_nested
   design/single_precision/adr/0001-reference-is-a-frozen-covariance
   design/single_precision/adr/0002-respect-fused-nesting-no-flattening
   design/single_precision/adr/0003-reference-delta-is-opt-in
   design/single_precision/adr/0004-timing-model-projection

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
