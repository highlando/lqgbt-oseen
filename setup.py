from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='lqgbt_oseen',
      version='2.0.0',
      description='LQGBT-controller for incompressible flows',
      license="GPLv3",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Heiland',
      author_email='jnhlnd@gmail.com',
      url="https://github.com/highlando/lqgbt-oseen",
      packages=['lqgbt_oseen'],  # same as name
      install_requires=['numpy', 'scipy',
                        'dolfin_navier_scipy==1.1.1',
                        'sadptprj_riclyap_adi==1.0.3',
                        'distributed_control_fenics==1.0.0'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          ]
      )
