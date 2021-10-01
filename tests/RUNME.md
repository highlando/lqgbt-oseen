Performance Regions
---

Code and instructions to reproduce the figures of the paper.


**Optional**: Consider setting the number of threads to a decent number.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Run the Simulations

```sh
source script-numtests-cylwake-minimal.sh  # test runs for the cylinder wake (reduced setup)
source script-numtests-cylwake.sh  # test runs for the cylinder wake
source script-numtests-dbrotcyl.sh  # test runs for the double cylinder
```

## Check for Stability

```sh
cd visualization
python3 check-simu-data.py --cw --showplots
python3 check-simu-data.py --dbrc --showplots
```

## Plot the Results

```sh
cd visualization
pdflatex minilat.tex
```

## Data and Directories

 * `simudata` -- results of the simulations -- parameter `ddir` in `compscript.py`
 * `cachedata` -- directory for data caching
 * `testdata` -- provides the raw data (the Gramians) for the simulation
   * get them from the [Zenodo repository](https://doi.org/10.5281/zenodo.4507759)
   * or from the Matlab runs --> see Zenodo
 * `plots` -- paraview plots  -- parameter `prvdir` in `compscript.py`
 * `visualization` -- post processing happens here
 * `mesh` -- the meshes for the simulation

## Troubleshooting

For *python3.5* an elder version of *NumPy* is needed. For that one may set 

```python
numpy==1.18.0,
```

in `../setup.py` before the installation.
