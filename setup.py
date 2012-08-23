import ez_setup
ez_setup.use_setuptools()

import os
import sys
from setuptools import setup

version_py = os.path.join(os.path.dirname(__file__), 'linregress', 'version.py')
version = open(version_py).read().strip().split('=')[-1].replace('"','')

long_description = """
Package for managing linear regression in R, via Python (and Rpy2)
"""

setup(
        name="linregress",
        version=version,
        install_requires=['numpy', 'rpy2'],
        packages=['linregress',
                  'linregress.test',
                  'linregress.test.data',
                  #'linregress.scripts',
                  ],
        author="Ryan Dale",
        description=long_description,
        long_description=long_description,
        url="none",
        package_data = {'linregress':["test/data/*"]},
        package_dir = {"linregress": "linregress"},
        #scripts = ['linregress/scripts/example_script.py'],
        author_email="dalerr@niddk.nih.gov",
        classifiers=['Development Status :: 4 - Beta'],
    )
