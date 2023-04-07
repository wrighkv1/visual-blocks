# Visual Blocks for Google Colaboratory

Visual Blocks is a Python package for use within [Google Colaboratory][]
notebooks. 

[Google Colaboratory]: https://colab.research.google.com 

## For Users

`!pip install visual_blocks` and `import visual_blocks` in your Colaboratory
notebooks. See the example notebooks in the directory [examples/](examples/).

## For Developers

`!pip install git+https://...` in a notebook installs the package straight from
the latest, unreleased source in Github. The notebooks in the [tests/](tests/)
directory use this method.

The directory [scripts/](scripts/) contains turnkey scripts for common
developer tasks such as building and uploading the Python distribution package.
