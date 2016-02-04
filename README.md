# Goal-directed decision making with spiking neurons
Supplementary material for the paper "Goal-directed decision making with spiking neurons"

### Requirements
The scripts were tested on Linux and MacOS with the following software installed

- python 2.7.11
- matplotlib 1.5.1
- numpy 1.10.2
- scipy 0.16.1
- cython 0.23.4

### Installation
For faster execution some functions have been written in Cython and need to be compiled by running in the directory 'code':
`python setup.py build_ext --inplace`

To clean up temporary files follow it by:
`python setup.py clean --all`

### Execution
The scripts are in subfolders of 'code' with names obvious from the paper. For the benchmark tasks from the machine learning literature running the scripts in their folder (blackjack, maze, pendulum) produces and saves the figures in the same folder. Because the computation takes some time (hours for pendulum, even with compiled code) the results for producing the figures are in the repo and read in if available else computed.
The other scripts can be run with `python script.py` to show the figures during code execution, or with any argument e.g. `python script.py 1` to save them to disk. 
