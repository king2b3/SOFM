# Bayley King
# Python 3.7.3
# Function library to be used with SOFM network
# I'll comment through the rest of the code a little more later
# For now this is just a working func library 

def graphHeatmap(Input,Output):
    '''
    Function to graph the final heatmap for the best input for each output neuron
    '''
    import matplotlib.pyplot as plt
    import pickle as pkl
    plt.figure()
    h = []
    path1 = 'SavedWeights/'+Input
    path2 = 'SavedWeights/'+Output
    output = pkl.load( open(path1, "rb" ) )
    for x in range(0, len(output), 10):  
        h.append(output[x:x + 10])
    import seaborn as sns; sns.set()
    ax = sns.heatmap(h, annot=True,xticklabels=False,yticklabels=False,cbar=False)
    #plt.show()
    plt.savefig(path2)


def threshWeights(File,layers,thresh):
    '''
    Returns the saved weights with a threshold value set to create a clearer map
    '''
    import numpy as np
    import csv
    weights = np.random.randn(layers[1],layers[0])
    path = 'SavedWeights/'+File
    dataFile = open(path)
    lines = dataFile.readlines()
    dataFile.close()
    counter = 0
    for neuron in range(layers[1]):
        for w in range(layers[0]):
            weights[neuron][w] = lines[counter]
            counter += 1
    
    for neuron in range(layers[1]):
        for w in range(layers[0]):
            if weights[neuron][w] < thresh:
                weights[neuron][w] = 0
            else:
                weights[neuron][w] = 1
            counter += 1

    return weights    


def loadWeights(File,layers):
    '''
    Returns the saved weights
    '''
    import numpy as np
    import csv
    weights = np.random.randn(layers[1],layers[0])
    path = 'SavedWeights/'+File
    dataFile = open(path)
    lines = dataFile.readlines()
    dataFile.close()
    counter = 0
    for neuron in range(layers[1]):
        for w in range(layers[0]):
            weights[neuron][w] = lines[counter]
            counter += 1
    return weights


def lookAtTheseBoys(num,weights,size):
    '''
    Recursive function call to help with plotting for weight map
    '''
    import numpy as np
    return (np.transpose(weights[num].reshape(size,size)))

def weightPlot(weights,Output):
    '''
    Plots the weight map of all the output neurons
    Its slightly hardcoded to work for a 10x10 output map, but will be changing this later
    Just need to change the inntances of the int 10 below to the appropiate sizes
    '''
    import matplotlib.pyplot as plt
    plt.figure()
    numfella = 0
    f, axarr = plt.subplots(10,10) #,constrained_layout=True)#,gridspec_kw = {'wspace':0, 'hspace':0})
    for row in range(10):
        for col in range(10):
            axarr[row,col].imshow(lookAtTheseBoys(numfella,weights,28),cmap=plt.get_cmap('gray_r'))
            #axarr[row,col].grid('on', linestyle='--')
            axarr[row,col].set_xticklabels([])
            axarr[row,col].set_yticklabels([])
            axarr[row,col].set_aspect('equal')
            axarr[row,col].axis('off')
            numfella += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    path = 'SavedWeights/'+Output
    plt.savefig(path)
    #plt.show()

def loadMnist():
    '''
    function to load MNIST from their respective text files
    Will return a training and a testing set as a list.
    Each input in every list is a list in itself with the first list value being the 
    784 input pixel image of the digit, and the second value of the list being the numeric label.
    ie. [[0,0,0,0,.4,.34,.23, .... ,0,0],[1]] 
    '''
    import csv
    import random
    images = []
    with open('DataSets/MNISTnumImages5000.txt') as csv_file:
        lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter='\t')
        for row in lines:
            images.append(list(row))
        csv_file.close()
    labels = []
    with open('DataSets/MNISTnumLabels5000.txt') as csv_file:
        lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter='\n')
        for row in lines:
            labels.append([int(row[0])])
        csv_file.close()

    data = list(zip(images,labels))
    train = data[:4500]
    test = data[500:]
    random.shuffle(train)
    test.sort(key=sortSecond)
    return train,test

def sortSecond(val): 
    # Simple recurvsive function
    return val[1] 

def plotMetrics(max_epochs,Metrics,Output,tau,tauN,no,sigmaP):
    '''
    plots the %decrease in the neighborhood rate and the learning rate with the 
    average distance between the 1st and 2nd BMU against epochs
    '''
    S = []
    L = []

    for epochs in range(max_epochs):
        S.append(sigma(epochs,tauN,sigmaP)/sigmaP)
        L.append(decay_LR(epochs,tau,no)/no)


    import matplotlib.pyplot as plt
    import pickle as pkl

    #plt.figure()
    fig, ax1 = plt.subplots()
    epochs = range(max_epochs)
    path1 = 'SavedWeights/'+Metrics
    path2 = 'SavedWeights/'+Output
    metrics = pkl.load(open(path1, "rb" ))

    a = ax1.plot(epochs, metrics, 'ro',label='Average Distance')
    ax1.set_xlabel('Epohcs')
    ax1.set_ylabel('Average distance')
    ax1.set_ylim([0,4])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('"%" compared to initial value')
    b = ax2.plot(epochs, S, 'bx',label="Sigma Decay")
    c = ax2.plot(epochs, L, 'gx',label="LR decay")
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.title('Average distance between 1st and 2nd winning neuron over each epoch')
    ax1.legend()

    # added these three lines
    lns = a+b+c
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(path2)


def plotNeuronMap(weights,Output):
    '''
    Unsued function, see graphHeatmap() for new version
    '''
    import matplotlib.pyplot as plt
    plt.figure()
    numfella = 0
    f, axarr = plt.subplots(5,2) #,constrained_layout=True)#,gridspec_kw = {'wspace':0, 'hspace':0})
    for row in range(5):
        for col in range(2):
            axarr[row,col].imshow(lookAtTheseBoys(numfella,weights,28),cmap=plt.get_cmap('gray_r'))
            #axarr[row,col].grid('on', linestyle='--')
            axarr[row,col].set_xticklabels([])
            axarr[row,col].set_yticklabels([])
            axarr[row,col].set_aspect('equal')
            axarr[row,col].axis('off')
            numfella += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    path = 'SavedWeights/'+Output
    plt.savefig(path)
    #plt.show()


def decay_LR(t,tau,no):
    # copy of function from networks.py for testing
    import numpy as np
    a = no*np.exp(-t/tau)
    if a < .001:
        a = .001
    return a

def sigma(e,tauN,sigmaP):
    # copy of function from networks.py for testing
    import numpy as np
    s = sigmaP*np.exp(0-(e/tauN))
    if s < 2:
        s = 2
    return s
