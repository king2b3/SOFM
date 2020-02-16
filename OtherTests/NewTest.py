#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 7 2020

####### Decerations #######
from network import Network 
#import matplotlib.pyplot as plt
import funcs
###########################

training, test = funcs.loadMnist()
max_epochs = 15
no = 2
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 100
#batch_size = 50
trainBool = True
layers = [784,100]

NN = Network(layers)

trainBool = NN.train(training,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,trainBool)
NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layers[-1])
NN.saveWeights()

# Testing, kinda

weight = funcs.loadWeights()

funcs.plotMetrics(max_epochs)
funcs.graphHeatmap()
funcs.weightPlot(weight)