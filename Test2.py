#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 13 2020

'''
Long version testing of the SOM with MNIST, ran on dracarys
'''
####### Decerations #######
from network import Network 
import funcs
###########################

train,test = funcs.loadMnist()

####### Network Decleration #######
max_epochs = 900
no = 0.9
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 10
#batch_size = 50
trainBool = True
best_met = 9999
layers = [784,100]

for max_epochs in [100,450,2000]:
    for no in [.1,1,4]:
        for tau in [max_epochs,max_epochs/2,max_epochs/10]:
            for tauN in [max_epochs,max_epochs/2,max_epochs/10]:
	        	for sigmaP in [10,50,100]:
	            	trainBool = True
	            	NN = Network(layers)
	            	trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights
	            	if NN.metricsDistance[-1] < best_met:
	                	best_met = NN.metricsDistance[-1]
	                	NN.saveWeights()
	                	NN.saveMetrics(max_epochs,no,tau,tauN,sigmaP,layer)
	                	NN.test(test,trainBool)

###################################
