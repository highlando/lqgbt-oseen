Performance Regions
---

Code and instructions to reproduce the figures of the paper.

## Run the Simulations

```sh
./script-numtests-cylwake.sh  # test runs for the cylinder wake
./script-numtests-dbrotcyl.sh  # test runs for the double cylinder
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
 * `plots` -- paraview plots  -- parameter `prvdir` in `compscript.py`
 * `visualization` -- post processing happens here
 * `mesh` -- the meshes for the simulation

## Troubleshooting

For *python3.5* an elder version of *NumPy* is needed. For that one may set 

```python
numpy==1.18.0,
```

in `../setup.py` before the installation.
