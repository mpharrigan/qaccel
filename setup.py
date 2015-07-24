"""Set up qaccel."""

import os

from setuptools import setup, find_packages

from Cython.Build import cythonize


REFDIR = 'qaccel/reference/data'
if not os.path.exists(REFDIR):
    print("Making reference data")
    import make_reference_data

    make_reference_data.make_reference_data(REFDIR)

setup(name='qaccel',
      version='0.2',
      author='Matthew Harrigan',
      packages=find_packages(),
      zip_safe=False,
      ext_modules = cythonize(["qaccel/count.pyx"]),
      package_data={'qaccel': ['reference/*.*',
                               'reference/data/*.*']})
