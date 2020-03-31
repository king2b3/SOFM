# MNIST in SOFM

---

The code is for the most part pretty modularized. 

1. network.py contains the structure of the SOFM network. You can consturct the network object in a seperate controller file, like [MNIST.py]
    2. MNIST.py is 



If you are reading this Sid or Andy, the network.py file contains all of the code related to the SOFM, and should be generic to any data set.

I should have the comments in each function explaining their necessary inputs and what they do.

MNISTSOFM.py acts as the controller code, and reads in the MNIST data and calls the necessary functions from network.py. The network is also initialized in this code, so any changes you want in the network parameters should be found here. If you wanted to run a set of simulations, you could set a few for loops that iteratre through a list of parameters outside of the network decleration comments to have some auto testing.

### Branches

The master branch contains my original SOFM made for homework 5 in intillgent systems, and is set to work on the animal datasets that we used. 

The MNIST branch is my current working framework for the MNIST data set. The code is stable and will run but is serving as a backup currently. I'll merge the MNIST branch upon completion of the final network. 