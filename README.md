# Discovery
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17711453.svg)](https://doi.org/10.5281/zenodo.17711453)

<img src="discovery.png" alt="Logo" width="300" align="left" style="margin-right: 20px;"/>

_Discovery_ is a next-generation pulsar-timing-array data-analysis package, _built
for speed_ on a [JAX](https://jax.readthedocs.io/en/latest/) backend that supports
GPU execution and autodifferentiation.

<br clear="left"/>

## Installation

```bash
git clone https://github.com/nanograv/discovery.git
cd discovery
pip install -e .
```

_Discovery_ needs a modern Python with `numpy`, `scipy`, `jax`, and `pyarrow`. It
runs on CPU and is happiest on an Nvidia GPU with CUDA-enabled JAX. Some
subpackages (e.g. `discovery.samplers`, `discovery.flow`) and the docs build need
extra dependencies — install those with extras, e.g. `pip install -e ".[docs]"`.

## Documentation

Full documentation — guide, tutorials, the model cookbook, and the API
reference — is at **https://nanograv.github.io/discovery/**.

Graph-backend (metamatrix) notes for users and developers live in the Sphinx
docs under *Development* (`docs/metamatrix.md`, `docs/metamatrix_dev.md`).
Durable design notes (ADRs, research math, deletion checklist) are under
`docs/design/`.
