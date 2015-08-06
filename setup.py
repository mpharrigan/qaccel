"""Set up qaccel."""

import os
import versioneer

from setuptools import setup, find_packages
from Cython.Build import cythonize

REFDIR = 'qaccel/reference/data'
if not os.path.exists(REFDIR):
    print("Making reference data")
    import make_reference_data

    make_reference_data.make_reference_data(REFDIR)

setup(name='qaccel',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Matthew Harrigan',
      packages=find_packages(),
      zip_safe=False,
      ext_modules=cythonize(["qaccel/count.pyx"]),
      package_data={'qaccel': ['reference/data/*.*']}
      )
