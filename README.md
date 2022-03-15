# LumPy
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.org/gerritgr/LumPy.svg?branch=master)](https://travis-ci.org/gerritgr/LumPy)

Copyright: 2017, Gerrit Großmann, [Group of Modeling and Simulation](https://mosi.uni-saarland.de/) at [Saarland University](http://www.cs.uni-saarland.de/)

Version: 0.1 (Please note that this code is an experimental version in a very early development stage.)
## Overview
------------------
The LumPy toolset provides a proof of concept for lumping for DBMF/PA/AME equations for contact processes on complex networks.
It reduces the large number of ODEs given by the equation systems by clustering them and only solving a single ODE per cluster.
LumPy is written in Python 3 (requiring SciPy) and published under GPL v3 license.

As input, the tool takes model descriptions (containing degree distribution, contact and independent rules,
time horizon, etc.) and outputs the lumped (or original) equations in the form of a standalone python script. One can specify an arbitrary number of labels and rules.
## Installation
------------------
We recommend Python 3.5.2.
##### Requirements:
For extra fast clustering the [blist](https://pypi.python.org/pypi/blist) package is required, for extra fast symbolic computations [SymEngine](https://github.com/symengine/symengine).
[NetworkX](https://github.com/networkx/networkx) is needed to generate random graphs for numerical simulation.
[Imageio](https://github.com/imageio/imageio) can be used to create antinamted .gfis visualizing the ODE solutions for 2-State models,
[Tqdm](https://github.com/tqdm/tqdm) is a progress bar and [Pathos](https://pypi.python.org/pypi/pathos) is used for better multithreading.

Packages can be installed with
```sh
pip install -r requirements.txt
```
##### Use with Miniconda:
To use Lumpy in a Conda environment:
Fist, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then,
```sh
conda env create -f environment.yml
```

## Example Usage
-----------------
Typically, you run LumPy like this
```sh
python ame.py <modelfilepath>
```
which generates a python script (placed in the output folder) and executes it.
pa.py and dbmf.py may be used to create PA and DBMF equations, respectively.
Available options are:
```sh
positional arguments:
  model        path to modelfile

optional arguments:
  -h, --help   show this help message and exit
  --noautorun  generate code without executing it
  --nolumping  generate original equations without lumping
```
Optimal arguments overwrite the modelfile specification.



##### Caution:
* The code uses eval and exec, please use with sanitized input only.
* Existing files are overwritten without warning.
##### Output:
Ame.py outputs:

* the generated python script
* the lumped equations in a .csv file
* the original equations in a .csv file
* the full ODE solution as .csv file
* a visualization of the clustering

When the heuristic is used to determine cluster number, all intermediate steps are stored.
pa.py and dbmf.py only output the generated script and their respective solution.

## Files
------------------
| Filename | Function |
| ------ | ------ |
| ame.py | creates python code for lumped AME equations|
| pa.py | creates python code for lumped PA equations|
| dbmf.py | creates python code for lumped DBMF equations|
| simulation.py | performs (a very naive) MCMC on a model |
| model_parser.py | parses the model file and returns a dictionary containing model |
| utilities.py | useful functions regarding logging/timing/IO |
| LumpEngine.py | implements lumping/aggregation of equation |
| stopping_heuristic.py | heuristic for finding the correct number of clusters for AME|
| ClusterEngine.py | implements clustering for a given number of clusters for AME|
| DegreeClusterEngine.py | implements degree clustering, utilized in ClusterEngine.py and for PA/DBMF clustering |
| ExprGenerator.py | generates AME formulas |


## Model Descriptions
-----------------
(Will be replaced with YAML soon.)

The model files (placed in the model directory by default) specify the contact process, the network, initial fractions, the number of bins, etc. They are mostly self-explanatory, however, SIR.model contains a detailed description of the structure of a model file. An example SIR model file contains:
```
states = ["R", "I", "S"]                                      # states names are strings containing only a-z and A-Z, "#"" introduces a comment
degree_distribution = lambda x: x**(-2.5) if x > 0 else 0.0   # can be a function, list, or dictionary, for the AME zero-probabilities are not allowed
initial_distribution = {'S': 0.8, 'I': 0.1, 'R': 0.1}         # can be a function or dictionary
horizon = 0.5                                                 # time horizon
k_max = 100
R1: S+I -> I+I  with 0.005                                    # example of a contact rule
R2: I -> R with 3.0                                           # example of an independent rules
R3: R -> S with 8.0
bin_num = 25                                                  # DBMF/PA: number of clusters; AME: number of iter- and intra-degree clusters. Use auto to utilize heuristic.
```
## TODOs
------------------
*  Output C++/Julia(?) code instead of Python
*  Extensive parallelization during ODE step and for stopping heuristic
*  Use symbolic expressions (not strings) consequently during code generation
*  More consistency between PA/DBMF and AME computation
*  Stop using dicts in AME solver and delte unused betas
*  Refactor/clean/document code

## Known Issues and Pitfalls
------------------
* Underflows:
  Due to the size of the steps of the ODE solver, an ODE which converges to
  zero might become zero (or less than zero) and cause numerical problems.
  We solve this by truncating these values.
* There seems to be an issue with the blist package and the current Python version

## SIR Example
------------------
![Example](https://i.imgur.com/wQuYG21.png)

## More Information
------------------
on Lumping:

* Kyriakopoulos et al.
["Lumping of Degree Based Mean Field and Pair Approximation Equations for Multi-State Contact Processes"](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.012301)
* G. Großmann
"Lumping the Approximate Master Equation for Stochastic Processes on Complex Networks" (Master's thesis)

on AME:

* JP Gleeson
["High-accuracy approximation of binary-state dynamics on networks"](https://arxiv.org/pdf/1104.1537.pdf)

![LSP](http://25.media.tumblr.com/tumblr_mdwcwsB9Ji1rl3jgdo1_500.gif)

