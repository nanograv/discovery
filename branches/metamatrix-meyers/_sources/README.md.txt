# Discovery Documentation

This directory contains the Sphinx documentation for the Discovery project.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Clean Build Artifacts

```bash
cd docs
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation index
- `api/` - Auto-generated API reference documentation
- `_static/` - Static files (images, CSS, etc.)
- `_build/` - Build output (not tracked in git)

## Adding Documentation

The API documentation is automatically generated from docstrings in the source code. To add or update documentation:

1. Update docstrings in the Python source files using numpy docstring format
2. Rebuild the documentation with `make html`

### Numpy Docstring Format

Use numpy-style docstrings with sections like:

```python
def example_function(param1, param2):
    """
    Short description.

    Longer description with more details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    Notes
    -----
    Additional notes or mathematical equations:

    .. math::

        E = mc^2

    References
    ----------
    .. [1] Author et al. (Year), "Title", Journal.
    """
```

## Configuration

The documentation is configured in `conf.py` with:

- **Theme**: Read the Docs theme (`sphinx_rtd_theme`)
- **Extensions**:
  - `sphinx.ext.autodoc` - Auto-generate documentation from docstrings
  - `sphinx.ext.napoleon` - Support for numpy-style docstrings
  - `numpydoc` - Enhanced numpy docstring support
  - `sphinx.ext.mathjax` - Math equation rendering
  - `sphinx.ext.intersphinx` - Links to other documentation
