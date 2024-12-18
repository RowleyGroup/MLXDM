'''
Setup file
Setup the package under the name of torchanipbe0
'''

from setuptools import setup, find_packages

setup(name = 'torchanipbe0',
      version = '3.1.1',
      packages = find_packages(),
      include_package_data = True,
      install_requires = ['torch', 'requests','h5py', 'lark', 'ase'] )
