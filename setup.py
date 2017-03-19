from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='dcgan',
    version='0.1',
    author = 'myh1000',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
