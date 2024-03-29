from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='lqgbt_oseen',
      version='2.0.0',
      description='LQGBT-controller for incompressible flows',
      license="MIT",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/lqgbt-oseen",
      packages=['lqgbt_oseen'],  # same as name
      install_requires=['numpy', 'scipy',
                        'dolfin_navier_scipy',
                        'sadptprj_riclyap_adi',
                        'distributed_control_fenics'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT",
          "Operating System :: OS Independent",
          ]
      # external packages as dependencies
      )
