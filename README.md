# IN4325_IR
This is the reprodicibility project for Core_IR of IN4325 Information Retreival. The code in this project aims to reproduce a portion of [User Intent Prediction in Information-seeking Conversations](https://arxiv.org/pdf/1901.03489.pdf). 

# Dependencies
### Install
To install everything required to run this project a Makefile is given.
If your python 3 install location is at python3 use make install3
if your python 3 install location is at python use make install

This will install all required packages and download all required corpuses

### Run
If everything from the install script is installed the project can be run using the command make run or make run3, make run works if the python 3 command on your system is python, make run3 works if the python 3 command on your system is python3.

### Download Dataset

The dataset used in this project is the [MSDialog](https://ciir.cs.umass.edu/downloads/msdialog/) Dataset provided by the IN4325 team. Please add the dataset to the same directory as this project. If this is not the case, please change the datapaths accoordingly in *main.py*. 

### Download Features 

The **features** folder consists of a dump of the feature extractions conducted in this project. To speed up training, please download them in the same directory as the project. 

# To run project
Step 1: Download and upzip this repository into a directory
Step 2: Add the dataset to the same directory
Step 2: Run *main.py*

