#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 7 2020

####### Decerations #######
from network import Network 
import funcs
###########################

train,test = funcs.loadWeights()

####### Network Decleration #######
max_epochs = 400
no = 0.9
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 50
#batch_size = 50
trainBool = True
layers = [784,100]

NN = Network(layers)

trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
NN.test(test,trainBool)

#print('results for ',max_epochs,' max epochs')
###################################
