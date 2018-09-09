from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='lqgbt_oseen',
      version='v1.5',
      description='LQGBT-controller for incompressible flows',
      license="MIT",
      long_description=long_description,
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/lqgbt-oseen",
      packages=['lqgbt_oseen'],  # same as name
      install_requires=['numpy', 'scipy']  # external packages as dependencies
      )
