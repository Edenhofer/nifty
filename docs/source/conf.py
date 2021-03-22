import nifty7

extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',  # Render math as images
    'sphinx.ext.viewcode'  # Add links to highlighted source code
]
master_doc = 'index'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

project = u'NIFTy7'
copyright = u'2013-2021, Max-Planck-Society'
author = u'Martin Reinecke'

release = nifty7.version.__version__
version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False

html_theme = "sphinx_rtd_theme"
html_logo = 'nifty_logo_black.png'

exclude_patterns = [
    'mod/modules.rst', 'mod/nifty7.git_version.rst', 'mod/nifty7.logger.rst'
]
