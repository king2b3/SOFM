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
SimParm = "BestParam.txt"
Metrics = "metrics.p"
MapPickle = "map.p"
Weights = "Weights.txt"
OutputMap = "Map.tiff"
OutputW =  "Weights.tiff"
OutputMet = "Metrics.tiff"
#OutputN1 = "Neruon 83.tiff"
#OutputN2 = "Neruon 2.tiff"


NN = Network(layers)

trainBool = NN.train(training,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,MapPickle,trainBool)
NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layers[-1],Metrics,SimParm)
NN.saveWeights(Weights)

# Testing, kinda

weight = funcs.loadWeights(Weights,layers)
funcs.plotMetrics(max_epochs,Metrics,OutputMet,tau,tauN,no,sigmaP)
funcs.graphHeatmap(MapPickle,OutputMap)
funcs.weightPlot(weight,OutputW)
#funcs.plotNeuronMap(NN.neuronTest1,OutputN1)
#funcs.plotNeuronMap(NN.neuronTest2,OutputN2)


#print(NN.neuronTest)
#print(NN.weights[81])
