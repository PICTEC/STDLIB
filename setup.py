#-*-coding:utf-8;-*-
from setuptools import setup

def readme():
    with open('README.md','r') as f:
        return f.read()

setup(name='picstdlib',
      version='0.1',
      description='Helper package for signal/speech enhancement/processing',
      long_description=readme(),
      url='http://gitlab.com/pictec/picsignal',
      author='Paweł Tomasik',
      author_email='pawel.tomasik@pictec.eu',
      license=None,
      packages=['picsignal'],
      zip_safe=False,
      install_requires=['numpy', 'librosa', 'scipy', 'keras'],
      test_suite="nose.collector",
      tests_require=['pep8', 'nose>=1.0'],
      setup_requires=['nose>=1.0'])
