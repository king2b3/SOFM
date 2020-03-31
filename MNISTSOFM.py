#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 7 2020

####### Decerations #######
from network import Network 
import funcs
import ploting
import metrics
from params import *
import pickle as pkl
import csv
###########################

train,test = funcs.loadMnist()

NN = Network(layers)

trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
print('saving weights')
NN.saveWeights(Weights)
NN.test(test,MapPickle,trainBool,BMUPickle,Weights)
print('saving metrics')
NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layers[-1],Metrics,SimParm)
# Testing, kinda
#weight = funcs.threshWeights(Weights,layers,0.4)
weight = funcs.loadWeights(Weights,layers) # uncomment to plot true weight map
print('ploting metrics')
ploting.plotMetrics(max_epochs,Metrics,OutputMet,tau,tauN,no,sigmaP)
ploting.graphHeatmap(MapPickle,OutputMap)
ploting.weightPlot(weight,OutputW)

BMU = pkl.load(open(BMUPickle, "rb" ))
metrics.genTestTrustworthiness(BMU,OutNN)
# the files needs the testing points without labels, the next lines read it in
images = []
with open('DataSets/MNISTnumImages5000.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter='\t')
    for row in lines:
        images.append(list(row))
    csv_file.close()
metrics.genTestTrustworthiness(images,TestNN)
metrics.outputBMU(OutNN,TestNN)
