# Visual Blocks for Google Colaboratory

Visual Blocks is a Python package for use within [Google Colaboratory][]
notebooks. 

[Google Colaboratory]: https://colab.research.google.com 

## For Users

`!pip install visual_blocks` and `import visual_blocks` in your Colaboratory
notebooks. See the example notebooks in the directory [examples/](examples/).

## For Developers

The directory [scripts/](scripts/) contains turnkey scripts for common
developer tasks such as building the Python distribution package.

Building the Python distribution package requires the Python package `build`.
To install this dependency, run:

    python3 -m pip install --upgrade build
