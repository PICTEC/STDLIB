#-*-coding:utf-8;-*-
from setuptools import setup

def readme():
    with open('README.md','r') as f:
        return f.read()

setup(name='picstdlib',
      version='0.1',
      description='Set of utilities, tools and models installable',
      long_description=readme(),
      url='http://github.com/PICTEC/STDLIB',
      author='Paweł Tomasik',
      author_email='pawel.tomasik@pictec.eu',
      license='MIT',
      packages=['picsignal', 'picml', 'picutils'],
      zip_safe=False,
      install_requires=['numpy', 'librosa', 'scipy', 'keras', 'google-api-python-client', 'oauth2client'],
      test_suite="nose.collector",
      tests_require=['pep8', 'nose>=1.0'],
      setup_requires=['nose>=1.0'],
      include_package_data=True)
