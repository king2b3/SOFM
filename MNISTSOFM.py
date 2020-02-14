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
max_epochs = 12
no = 0.9
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 10
#batch_size = 50
trainBool = True
layers = [784,100]

NN = Network(layers)

trainBool = NN.train(training,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,trainBool)
NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layers[-1])
NN.saveWeights()


'''
####### Network Decleration #######
max_epochs = 900
no = 0.9
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 10
#batch_size = 50
trainBool = True
layers = [784,400]

NN = Network(layers)

trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,trainBool)

#print('results for ',max_epochs,' max epochs')
###################################
'''