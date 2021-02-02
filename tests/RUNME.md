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
cd data
python3 

## Plot the Results

## Data and Directories

 * `cachedata` -- directory for data caching
 * `data` -- results of the simulations -- parameter `ddir` in `compscript.py`
 * `plots` -- paraview plots  -- parameter `prvdir` in `compscript.py`
 * `visualization` -- postprocessing happens here
