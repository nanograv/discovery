Pulsar Data
===========

Discovery uses lightweight ``Pulsar`` objects saved as Arrow Feather files for efficient data storage and loading.

Data Format
-----------

Pulsar data is stored in the `Apache Arrow Feather format <https://arrow.apache.org/docs/python/feather.html>`_, which provides:

- Fast read/write performance
- Efficient columnar storage
- Cross-language compatibility
- Minimal memory overhead

Each Feather file contains:

- Time-of-arrival (TOA) data
- Observing frequencies
- Measurement uncertainties
- Design matrix (timing model)
- Pulsar position and metadata
- Optional: noise parameter defaults

Converting from Enterprise
--------------------------

If you have existing `Enterprise <https://github.com/nanograv/enterprise>`_ ``Pulsar`` objects,
you can convert them to Discovery format:

.. code-block:: python

   import discovery as ds
   import enterprise

   # Load Enterprise pulsar
   ent_psr = enterprise.Pulsar(parfile, timfile)

   # Convert and save as Feather file
   ds.Pulsar.save_feather(ent_psr, 'B1855+09.feather', noisedict)

The ``noisedict`` parameter is optional and allows you to include default noise parameter
values (e.g., EFAC, EQUAD values) in the Feather file.

Loading Pulsar Data
-------------------

Loading a Discovery pulsar is straightforward:

.. code-block:: python

   import discovery as ds

   # Load single pulsar
   psr = ds.Pulsar.read_feather('data/v1p1_de440_pint_bipm2019-B1855+09.feather')

   # Load multiple pulsars
   import glob
   psrs = [ds.Pulsar.read_feather(f)
           for f in glob.glob('data/v1p1_de440_pint_bipm2019-*.feather')]

Pulsar Object Attributes
-------------------------

A Discovery ``Pulsar`` object contains:

**TOA Data**
   - ``psr.toas`` - Times of arrival (MJD, in seconds)
   - ``psr.residuals`` - Timing residuals (seconds)
   - ``psr.toaerrs`` - TOA uncertainties (seconds)

**Observing Information**
   - ``psr.freqs`` - Observing frequencies (MHz)
   - ``psr.backends`` - Backend/receiver identifiers
   - ``psr.flags`` - Additional observational flags

**Timing Model**
   - ``psr.Mmat`` - Design matrix for timing model
   - ``psr.pos`` - Pulsar sky position (unit vector)

**Solar System**
   - ``psr.planetssb`` - Planet positions (solar system barycenter)
   - ``psr.sunssb`` - Sun position (solar system barycenter)

**Metadata**
   - ``psr.name`` - Pulsar name (e.g., 'B1855+09')
   - ``psr.noisedict`` - Dictionary of noise parameters (if saved)

Example Datasets
----------------

Discovery includes example datasets based on NANOGrav 15-year data in the ``data/`` folder:

.. code-block:: bash

   data/
   ├── v1p1_de440_pint_bipm2019-B1855+09.feather
   ├── v1p1_de440_pint_bipm2019-B1937+21.feather
   ├── v1p1_de440_pint_bipm2019-J0030+0451.feather
   └── ...

These can be used for testing and development:

.. code-block:: python

   # Load example pulsar
   psr = ds.Pulsar.read_feather('data/v1p1_de440_pint_bipm2019-B1855+09.feather')

   print(f"Pulsar: {psr.name}")
   print(f"TOAs: {len(psr.toas)}")
   print(f"Span: {(psr.toas.max() - psr.toas.min()) / 365.25 / 86400:.1f} years")

Working with Pulsar Data
-------------------------

Accessing residuals:

.. code-block:: python

   # Residuals as numpy array
   res = psr.residuals

   # Can be used directly in likelihood
   signals = [res, ...]

See Also
--------

- :doc:`/tutorials/simulations` - Simulating pulsar timing data
- :doc:`/api/pulsar` - Pulsar class API reference
