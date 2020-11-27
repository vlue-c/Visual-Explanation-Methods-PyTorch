from setuptools import find_packages
from setuptools import setup

setup(
    name='torchvex',
    version='0.1',
    author='hslee',
    url=None,
    description='Visual explanation methods for PyTorch.',
    packages=find_packages(), # exclude()
)
