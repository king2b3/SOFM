from network import Network 
import csv
import numpy as np

max_epochs = 4000
no = 0.1
tau = 2000
tauN = 750
sigmaP = 50
trainBool = True


layers = [29,100]
NN = Network(layers)



train = []
with open('DataSets/trainingSet.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in lines:
        train.append(list(row))
#print(train)
train = np.array(train)

testNum = []
#with open('testing1Set.txt') as csv_file:      # swap next line for problem 2
with open('DataSets/testing1Set.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in lines:
        testNum.append(row)

testLabel = []
#with open('testing2SetLabels.txt') as csv_file:  # swap next line for problem 2
with open('DataSets/testing1SetLabels.txt') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    for row in lines:
        testLabel.append(row)

test = list(zip(testNum,testLabel))
test = np.array(test)

trainBool = NN.train(train,max_epochs,no,tau,tauN,sigmaP,trainBool)

NN.test(test,trainBool)

