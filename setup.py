#-*- coding: utf-8 -*-
#File: setup.py
#Author: yobobobo(zhouboacmer@qq.com)

import sys
import os
import re
from setuptools import setup
def _find_packages(prefix=''):
  packages = []
  path = '.' 
  prefix = prefix
  for root, _, files in os.walk(path):
    if '__init__.py' in files:
      packages.append(
        re.sub('^[^A-z0-9_]', '', root.replace('/', '.'))
      ) 
  return packages

setup(
    name='tensorgo',
    version=0.1,
    author="zhoubo01",
    packages=_find_packages(__name__),
    package_data={'': ['*.so']}
)
