# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "discovery"
copyright = "2025, Michele Vallisneri"
author = "Michele Vallisneri"
release = "0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx extensions
extensions = [
    "autoapi.extension",  # auto api docs
    "sphinx.ext.viewcode",  # add a button to docs to jump to source code
    "sphinxcontrib.mermaid",  # enable mermaid diagrams
    "sphinx.ext.napoleon",  # allow google and numpy docstrings
    "sphinx.ext.autodoc.typehints",  # enable typehints
    "sphinx.ext.intersphinx",  # enable intersphinx for cross-site linking
    "sphinx_copybutton",  # add a button to copy code blocks
    "myst_nb",  # handles markdown and ipynb parsing
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "html_image",  # parse images included with html
    "smartquotes",  # automatically add open an close quote styles
    "dollarmath",  # use typical $$ and $ math delimeters
    "amsmath",  # render amsmatch environments
    "linkify",  # make things clickable links if they should be
]

# When rendering ipynb or md, automatically generate anchors up to 3 levels
# deep
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#auto-generated-header-anchors
myst_heading_anchors = 3

# Do not execute notebooks when rendering
nb_execution_mode = "off"

# Common things to exclude from builds
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Add mappings for intersphinx
intersphinx_mapping = {
    "enterprise": ("https://enterprise.readthedocs.io/en/latest/", None),
    "enterprise_extensions": ("https://enterprise-extensions.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# Let autoapi know where our code lives
autoapi_dirs = ["../src/", "../tests/"]

# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
autoapi_options = [
    "members",  #  Display children of an object
    "undoc-members",  #  Display objects that have no docstring
    "private-members",  #  Display private objects (eg. _foo in Python)
    "show-inheritance",  # Display a list of base classes below the class signature.
    "show-module-summary",  # Include autosummary directives in generated module documentation.
    "special-members",  # Display special objects (eg. __foo__ in Python)
    "imported-members",  # Display objects imported from the same top level package or module.
    "inherited-members",  #  Display children of an object that have been inherited from a base class.
]
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_typehints
autodoc_typehints = "description"
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-templates_path
templates_path = ["_templates"]
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# For the logo to also show in the README
html_extra_path = ["../discovery.png"]

# https://pydata-sphinx-theme.readthedocs.io/en/stable/
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Let users suggest doc improvements
html_theme_options = {
    "use_edit_page_button": True,
}

# metadata
html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "nanograv",
    "github_repo": "discovery",
    "github_version": "main",
    "doc_path": "docs/",
}
