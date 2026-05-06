# Sphinx configuration for Ocean Flow Matching documentation

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# This allows `import src` and therefore the project modules to be imported by
# the autodoc extension.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Ocean Flow Matching"
copyright = "2026, Chiara Zelco"
author = "Chiara Zelco"

# The full version, including alpha/beta/rc tags
release = "0.5"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Napoleon settings to support NumPy-style docstrings (our chosen style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx configuration ---------------------------------------------

enabled = True
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}
