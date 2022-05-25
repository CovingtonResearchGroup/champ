<p align="center>
<img src="https://github.com/CovingtonResearchGroup/CO2-speleogenesis/blob/spim/img/champ3.png" width="320" height="240">
                                                                                                                        </p>
# Champ - Channel morphodynamics in Python

Champ is a Python package to simulate bedrock channel morphology in a single cross-section or a reach defined by multiple cross-sections.

## Requirements and installation
Running a simulation with Champ requires NumPy, SciPy, PyYAML, and Cython. For visualization it requires Pillow, matplotlib, and Mayavi. The required packages are available in the default Anaconda Individual Edition installation.

The package is installed from a console with:
```bash
python setup.py install
```

## Usage

* The main entry point for the code is sim.py, which contains the simulation object.

* runSim.py contains code for setting up and running simulations from yaml files that specify parameters. A set of simulations can be run in parallel by running run_yml_dir.py on a directory containing multiple yaml files.

* Code related to cross-section geometrical and flow calculations is contained in crossSection.py

* Unit tests are included in the tests folder. They can be run from the command line with python -m pytest, assuming pytest is installed.

*Thanks to Amanda Anders for creating our Champ logo!*
