"""Set up qaccel."""

import os

from setuptools import setup, find_packages


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
      package_data={'qaccel': ['reference/*.*',
                               'reference/data/*.*']})
