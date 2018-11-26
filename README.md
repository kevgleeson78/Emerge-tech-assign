# Emerging Technologies Assignment
Author: [Kevin Gleeson](https://github.com/kevgleeson78)

Fourth year student at: [GMIT](http://gmit.ie) Galway

### This repository holds a python script and four jupyter notebooks.
* The python script creates a neural network that can read and recognise hand written digits.
 Please see the notebook in this repository for a further explanation.
* A notebook explaining the numpy random package and an example of some distributions that it uses.
* A notebook explaining the Iris data set and the difficulty that a neural network might have in classifying the different species of flower.
* A notebook explaining the mnist dataset and how to open and read the datasets gzipped files efficiently into memory.
* A notebook explaining each element of the digit recognition python script held in this repository.



## Cloning, compiling and running the notebooks and python script.

1. Download [git](https://git-scm.com/downloads) to your machine if not already installed.

2. Download and install [Anaconda](https://www.anaconda.com/download/) this will install all of the python packages and jupyter needed to run the files in this repository

3. Open git bash and cd to the folder you wish to hold the repository.
Alternatively you can right click on the folder and select git bash here.
This will open the git command prompt in the folder selected.
 
 4. To clone the repository type the following command in the terminal making sure you are in the folder needed for the repository.
```bash
>git clone https://github.com/kevgleeson78/Emerge-tech-assign.git
```
6. Open the repository folder you have cloned and create a new folder called "data"
5. Download the four files of the mnist data set [here](http://yann.lecun.com/exdb/mnist/) and save the files into the new data folder
### Running the notebooks
1. Cd to the cloned folder and run the command line. When the command line is open type 
```cmd
jupyter notebook
```
This will run jupyter and open the explorer in a browser.

From here you can open the notebooks by simply clicking on the notebook you wish to open.

2. The current state of the notebook is saved when first opened. You can clear the state and rerun all cells by goint to the top menu click "kernel > restart and run all"
This will run all of the cells in the notebook.

### Running the python script
1. Navigate to the cloned repository folder and open a command prompt.
2. In the command prompt type 
```
python digitrec.py
```
This will run the script and display the output within the command prompt window.



