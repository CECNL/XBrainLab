# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "../..")))
# make sure capture outer XBrainLab
# print(sys.path)



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XBrainLab'
copyright = '2023, CECNL'
author = 'CECNL'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"


extensions = [
    # builtin
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    # others
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autodoc_default_options = {"inherited-members": None}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
  # side bar setting
  "navigation_with_keys": False,
  "show_toc_level": 1,
  "secondary_sidebar_items": ["page-toc", "edit-this-page"],
  "article_header_start": [], # disable breadcrumbs
  # nav bar External link icon
  "icon_links_label": "External Links",
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CECNL/XBrainLab",
            "icon": "fa-brands fa-square-github",
        },],
  # nav bar setting
#   "logo": {
#         "text":"XBrainLab",
#         "alt_text": "Home",
#     }
}

html_static_path = ['_static']
html_logo = "_static/logo_title.svg"

# left out "Created using Sphinx" in footnote
html_show_sphinx = False

# special sidebar for mainpage
html_sidebars = {
    'index': ['globaltoc.html'],
}


bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""
