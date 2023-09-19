# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "../.."))) # make sure capture outer XBrainLab
# print(sys.path)



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XBrainLab'
copyright = '2023, CECNL'
author = 'CECNL'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
	'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []


autosummary_generate = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_sidebars = {
    '**': ['globaltoc.html', 'searchbox.html'] 
}