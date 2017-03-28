from setuptools import find_packages
from setuptools import setup

if __name__ == '__main__':
  setup(name='dcgan64', packages=['trainer'], install_requires=['google-cloud', 'pillow'])
