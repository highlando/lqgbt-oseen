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

 * `data` -- results of the simulations -- parameter `ddir` in `compscript.py`
 * `plots` -- paraview plots  -- parameter `prvdir` in `compscript.py`
 * `visualization` -- postprocessing happens here
 * `cachedata` -- directory for data caching
 * `testdata` -- provides the raw data (the Gramians) for the simulation
 * `mesh` -- the meshes for the simulation
