# Goal-directed decision making with spiking neurons
Supplementary material for the paper "Goal-directed decision making with spiking neurons"

### Installation
The scripts were tested on Linux and MacOS with the following software installed

- python 2.7.11
- matplotlib 1.5.1
- numpy 1.10.2
- scipy 0.16.1
- cython 0.23.4

For faster execution some functions have been written in Cython and need to be compiled by running in the directory 'code':
`python setup.py build_ext --inplace`

To clean up temporary files follow it by:
`python setup.py clean --all`
