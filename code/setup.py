# setup.py
# run with:         python setup.py build_ext --inplace
# clean up with:    python setup.py clean --all

from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy as np


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

setup(ext_modules=cythonize("*.pyx"),
      include_dirs=[np.get_include()])
with cd("pendulum"):
    setup(ext_modules=cythonize("*.pyx"),
          include_dirs=[np.get_include()])
with cd("2AFC"):
    setup(ext_modules=cythonize("*.pyx"),
          include_dirs=[np.get_include()])
