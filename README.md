# MNIST in SOFM

---

The code is for the most part pretty modularized. 

* [network.py](https://github.com/king2b3/SOFM/blob/MNIST/network.py) contains the structure of the SOFM network. You can construct the network object in a separate controller file, like [MNISTSOFM.py](https://github.com/king2b3/SOFM/blob/MNIST/MNISTSOFM.py).
* [MNISTSOFM.py](https://github.com/king2b3/SOFM/blob/MNIST/MNISTSOFM.py) or [ECG.py](https://github.com/king2b3/SOFM/blob/MNIST/ECG.py) acts as a controller, and contains the constructor for the network. It also controls the testing of the trained network, plotting figures, calculating metrics, and saving the trained network. 
* [funcs.py](https://github.com/king2b3/SOFM/blob/MNIST/funcs.py) acts as the general function library the this project uses. It contains functions to load in the data sets, thresholding weights, etc.
* [plotting.py](https://github.com/king2b3/SOFM/blob/MNIST/plotting.py) serves as the function library for plotting various features of the network.
* [metrics.py](https://github.com/king2b3/SOFM/blob/MNIST/metrics.py) serves as the function library for generating metrics of the trained network.
* [DataSets](https://github.com/king2b3/SOFM/tree/MNIST/DataSets) contains the data sets that we have used with the network. Some of them are in their basic form, others are zipped due to their size.
* [params.py](https://github.com/king2b3/SOFM/tree/MNIST/params.py) contains the variable initialization of the trained network. each input / output file is defined here along with the parameters of the network. 

### The general flow of your controller could be as follows
1. construct [network.py](https://github.com/king2b3/SOFM/blob/MNIST/network.py) and load data set of your choice. a few sample data sets are located in [funcs.py](https://github.com/king2b3/SOFM/blob/MNIST/funcs.py)
2. train the network using the network class. you can define your own parameters or just use the [params.py](https://github.com/king2b3/SOFM/tree/MNIST/params.py) structure. the train function will return a bool of False, which is used in the testing of the network.
3. save the weights, then test the network. this will generate and save the output map of the trained network according to the testing set. 
4. calculate and plot / output the metrics of the network. 

### Other usage
You don't need to use a controller for all use cases of the network. If you properly saved the last working version of the network, you can just set up your controller to output the metrics and look at the output plots. You don't need to train for every case. You should at the minimum test the network everytime you want to see any sort of output, just to make sure that the output map is properly stored. Since the training is the overhead, this shouldn't be much of an issue. 

### Branches

The master branch contains the current working version of the network. It might not be the most up to date version, but it should be a stable version.

The dev branch is my current working framework for the network. If you want to see what I am currently struggling with at the given time, look here. If you want a stable version of the network, I would look at the master branch