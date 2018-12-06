# adcs-simulation

This modeling and simulation software is used to characterize the behavior of attitude determination and control systems (ADCS) on spacecraft, and allows for rapid iteration to demonstrate the viability of a specific suite of sensor, actuator, and controller designs for a given spacecraft design. The mathematical models used for all modeling are described in the accompanying paper.

## Running the Project

### Python & Package Installation
This software has the following Python dependencies:
- numpy
- scipy
- matplotlib

There are two routes to installation:
1. Download and install the Python 3.7 distribution of [Anaconda](https://www.anaconda.com/download/) to avoid installing dependencies manually, as it will already have the above libraries installed (recommended for Mac/Linux, required for Windows).
2. Use the PyPi package manager to install the requirements listed in the **requirements.txt** file:
`pip install -r requirements.txt`

### Running on Mac/Linux
1. Download the software present in this repository.
2. Open a terminal and navigate to the directory containing **project.py** (in **adcs-simulation**).
3. Run the project with the command `python3 project.py`.

If there are issues with the dependencies not importing successfully, confirm that the `python3` command points to your Anaconda installation by running `which python3` (it should point to `~/anaconda/bin/python3`, where `~` is the user prefix). If it does not, you may try the alternative command: `~/anaconda/bin/python3 project.py`

### Running on Windows
1. Download the software present in this repository.
2. Open Anaconda prompt (installed above) and navigate to the directory containing **project.py** (in **adcs-simulation**).
3. Run the project with the command `python3 project.py`.


## Software Documentation
The Sphinx Python documentation generator was used to create HTML-based documentation for the code written here. For an overview of the project, it is suggested to start with the [modules](docs/modules.html) page (at docs/modules.html).


This repository is a product of the ASE 372K (Attitude Dynamics) course final project.
