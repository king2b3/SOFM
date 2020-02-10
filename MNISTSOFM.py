#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   Feb 7 2020

####### Libraries #######
import csv
import random
import numpy as np
from tabulate import tabulate
#########################

####### Classes #######
from network import Network 
#######################

####### Functions #######
def sortSecond(val): 
    return val[1] 
#########################

####### Load MNIST Data #######
images = []
with open('DataSets/MNISTnumImages5000.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter='\t')
    for row in lines:
        images.append(list(row))

labels = []
with open('DataSets/MNISTnumLabels5000.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter='\n')
    for row in lines:
        labels.append([int(row[0])])

data = list(zip(images,labels))
train = data[:4000]
test = data[4000:]
random.shuffle(train)
test.sort(key=sortSecond)
###############################

####### Network Decleration #######
max_epochs = 500
no = 0.1
tau = max_epochs/2
tauN = max_epochs/2
sigmaP = 50
trainBool = True

layers = [784,100]
NN = Network(layers)

#trainBool = NN.train(data,max_epochs,no,tau,tauN,sigmaP,trainBool) #Comment out line to run on saved weights

NN.test(test,trainBool)
###################################