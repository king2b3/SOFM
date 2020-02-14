#   Bayley King
#   SOFM Base Network 
#   Python 3.7.3

####### Libraries #######
import csv
import random
import numpy as np
#from tabulate import tabulate
import pickle as pkl
#########################

class Network(object):

    def __init__(self,layers):
        '''
        inits of all of the variable local to the object
        '''
        self.num_Layers = len(layers)
        self.layers = layers
        self.weights = np.random.randn(self.layers[1],self.layers[0])
        self.fullMap = []  
        [self.fullMap.append(['r',9999999]) for i in range(layers[1])]
        self.sq = int(np.sqrt(layers[1]))
        self.SOM_Shape = np.array([self.sq, self.sq])
        self.Index = np.mgrid[0:self.SOM_Shape[0],0:self.SOM_Shape[1]].reshape(2, self.SOM_Shape[0]*self.SOM_Shape[1]).T
        self.metricsDistance = []


    def winning_neuron(self,x):
        '''
        This function returns the index of the winning neuron
        '''
        c = np.linalg.norm(x[0] - self.weights, axis=1)
        a = np.argmin(c)
        c[a] = 9999999
        b = np.argmin(c)
        return a,b


    def update_weights(self, eta, sigma, N, x):
        '''
        This changes the weights based off of the neighboorhood
        function after finding the best performing neuron
        '''
        d = np.square(np.linalg.norm(self.Index - self.Index[N], axis=1))
        h = np.exp(-d/(2 * sigma**2))
        self.weights += eta * h[:, np.newaxis] * (x[0] - self.weights)


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

    def metrics(self,a,b):
        x1,y1 = np.divmod(a,self.sq)
        x2,y2 = np.divmod(b,self.sq)
        dist = np.sqrt(((x1-x2)**2 + (y1-y2)**2))
        return dist


    def sigma(self,e,tauN,sigmaP):
        '''
        This function will decay the sigma rate of the 
        neighboorhood function across each epoch
        '''
        return (sigmaP*np.exp(0-(e/tauN)))
    

    def decay_LR(self,t,tau,no):
        '''
        This function will decay the learning rate over the training set
        t is the current epoch of the system.
        '''
        a = no*np.exp(-t/tau)
        if a < .1:
            a = .1
        return a

    def train(self,training,max_epochs,no,tau,tauN,sigmaP,trainBool,batch_size = None):
        '''
        this function trains the network for a max number of epochs
        '''
        trainBool = False
        if batch_size is None:
            batch_size = len(training)
        print('##############')
        print('Starting Training')
        print('##############')
        for e in range(max_epochs):
            random.shuffle(training)
            eta = self.decay_LR(e,tau,no)
            sigma = self.sigma(e,tauN,sigmaP)
            batch = training[:batch_size]
            distance = []
            for i in batch:
                N,N2 = self.winning_neuron(i)
                self.update_weights(eta,sigma,N,i)
                distance.append(self.metrics(N,N2))
            # epoch complete
            self.metricsDistance.append(np.average(distance))
            #if e % 10 == 0:
            #print('epoch',e,'complete')
        return trainBool

    def saveWeights(self):
        with open('SavedWeights/Weights.txt', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for neuron in range(len(self.weights)):
                for w in range(len(self.weights[0])):
                    csv_writer.writerow([self.weights[neuron][w]])


    def saveMetrics(self,max_epochs,no,tau,tauN,sigmaP,layer):
        pkl.dump(self.metricsDistance, open("SavedWeights/metrics.p", "wb" ) )
        with open('SavedWeights/BestParam.txt', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Max epochs: '+str(max_epochs)])
            csv_writer.writerow(['Learning Rate: '+str(no)])
            csv_writer.writerow(['Tau: '+str(tau)])
            csv_writer.writerow(['TauN: '+str(tauN)])
            csv_writer.writerow(['SigmaP: '+str(sigmaP)])
            csv_writer.writerow(['Final layers 784',layer])


    def test(self,testing,trainBool):
        '''
        This function will test the trained network with the new
        data points
        '''
        if trainBool:
            print('Loading Weights......')
            dataFile = open('SavedWeights/Weights.txt')
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
        # Finds the winning neuron for each input, stores that in Win
        Win = []
        for t in testing:
            Win.append(self.winning_neuron(t[0])[0])
            self.test_winning_neuron(t)
        # Prints the best neuron for each input
        #for k in range(len(Win)):
        #    print(Win[k],testing[k][1][0])
        # Prints the best input for each neuron on map
        output = []
        for i in self.fullMap:
            output.append(i[0][0])
        h = []
        for x in range(0, len(output), self.sq):  
            h.append(output[x:x + self.sq])
        #print(tabulate(h))
        pkl.dump(output, open("SavedWeights/map.p", "wb" ) )
