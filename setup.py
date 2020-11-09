from setuptools import find_packages
from setuptools import setup

setup(
    name='torchex',
    version='0.1',
    author='hslee',
    url=None,
    description='Re-implemented explanation methods for PyTorch',
    packages=find_packages(), # exclude()
)
