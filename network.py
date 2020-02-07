#   Bayley King
#   Intelligent Systems Homework 5
#   Python 3.6.8
#   Due 12/05/19

####### Libraries #######
import csv
import random
import numpy as np
from tabulate import tabulate
#########################

class Network(object):

    def __init__(self,layers):
        '''
        inits of all of the variable local to the object
        '''
        self.num_Layers = len(layers)
        self.layers = layers
        self.weights = np.array([np.random.randn(x,y)  for x,y in zip(layers[1:],layers[:-1])])
        self.weights = self.weights[0]
        self.fullMap = []  
        [self.fullMap.append(['r',9999999]) for i in range(100)]
        self.SOM_Shape = np.array([10, 10])
        self.Index = np.mgrid[0:self.SOM_Shape[0],0:self.SOM_Shape[1]].reshape(2, self.SOM_Shape[0]*self.SOM_Shape[1]).T



    def winning_neuron(self,x):
        '''
        This function returns the index of the winning neuron
        '''
        return np.argmin(np.linalg.norm(x - self.weights, axis=1))


    def update_weights(self, eta, sigma, x):
        '''
        This changes the weights based off of the neighboorhood
        function after finding the best performing neuron
        '''
        i = self.winning_neuron(x)
        d = np.square(np.linalg.norm(self.Index - self.Index[i], axis=1))
        h = np.exp(-d/(2 * sigma**2))
        self.weights += eta * h[:, np.newaxis] * (x - self.weights)


    def test_winning_neuron(self,X):
        '''
        This function populates the Neurons with Maximal Response to each Animal
        and the Preferred Response for each Neuron charts 
        '''
        D = np.linalg.norm(X[0] - self.weights, axis=1)
        for i in range(len(D)):
            if D[i] < self.fullMap[i][1]:
                self.fullMap[i][0] = X[1]
                self.fullMap[i][1] = D[i]



    def sigma(self,e):
        '''
        This function will decay the sigma rate of the 
        neighboorhood function across each epoch
        '''
        return (sigmaP*np.exp(0-(e/tauN)))
    

    def decay_LR(self,t):
        '''
        This function will decay the learning rate over the training set
        t is the current epoch of the system.
        '''
        return (no*np.exp(-t/tau))


    def train(self,training):
        '''
        this function trains the network for a max number of epochs
        '''
        trainBool = False
        print('##############')
        print('Starting Training')
        print('##############')
        for e in range(max_epochs):
            eta = self.decay_LR(e)
            sigma = self.sigma(e)
            for i in training:
                N = self.winning_neuron(i)
                self.update_weights(eta,sigm = [29,100]a,i)
            # epoch complete
            if e % 100 == 0:
                print('epoch',e,'complete')
        with open('Weights.txt', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for neuron in range(len(self.weights)):
                for w in range(len(self.weights[0])):
                    csv_writer.writerow([self.weights[neuron][w]])
        return trainBool


    def test(self,testing,trainBool):
        '''
        This function will test the trained network with the new
        data points
        '''
        if trainBool:
            print('Loading Weights......')
            dataFile = open('Weights.txt')
            lines = dataFile.readlines()
            dataFile.close()
            counter = 0
            for neuron in range(len(self.weights)):
                for w in range(len(self.weights[0])):
                    self.weights[neuron][w] = lines[counter]
                    counter += 1
        

        print('##############')
        print('Starting Testing')
        print('##############')
        Win = []
        for t in testing:
            Win.append(self.winning_neuron(t[0]))
            self.test_winning_neuron(t)
        #for k in range(len(Win)):
        #    print(Win[k],testing[k][1][0])
        output = []
        for i in self.fullMap:
            output.append(i[0][0])
        h = []
        for i in range(0, len(outpu = [29,100]
        #for h in range(0,100,10):
        #    print(output[h][0],'\t',output[h+1][0],'\t',output[h+2][0],'\t',output[h+3][0],'\t',output[h+4][0],'\t',output[h+5][0],'\t',output[h+6][0],'\t',output[h+7][0],'\t',output[h+8][0],'\t',output[h+9][0])

max_epochs = 4000
no = 0.1
tau = 2000
tauN = 750
sigmaP = 50
trainBool = True


layers = [29,100]
NN = Network(layers)

train = []
with open('trainingSet.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in lines:
        train.append(list(row))
print(train)
train = np.array(train)

testNum = []
#with open('testing1Set.txt') as csv_file:      # swap next line for problem 2
with open('testing1Set.txt') as csv_file:
    lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    for row in lines:
        testNum.append(row)

testLabel = []
#with open('testing2SetLabels.txt') as csv_file:  # swap next line for problem 2
with open('testing1SetLabels.txt') as csv_file:
    lines = csv.reader(csv_file, delimiter=',')
    for row in lines:
        testLabel.append(row)

test = list(zip(testNum,testLabel))
test = np.array(test)

trainBool = NN.train(train)

NN.test(test,trainBool)

