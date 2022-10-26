#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages
import re

VERSIONFILE="semforage/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='semforage',
      version=verstr,
      description='Foraging methods for Semantic Memory Search',
      url='https://github.com/larryzhang95/forager',
      author='IU Cognitive Computing Laboratory',
      author_email='larzhang@iu.edu',
      keywords=['fluency', 'foraging','psychology','memory','cognitive modeling'],
      packages=['semforage'],
      include_package_data=True,
      install_requires=['numpy','scipy','more_itertools','pandas','tqdm','nltk','difflib','requests'],
      python_requires='>=3.8',
      zip_safe=False,
      classifiers=[
            'Programming Language :: Python :: 3.8'
      ]
      )