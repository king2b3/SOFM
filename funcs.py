# Bayley King
# Python 3.7.3
# General function library to be used with SOFM network


def threshWeights(File,layers,thresh):
    '''
    Returns the saved weights with a threshold value set to create a clearer map
    '''
    import numpy as np
    import csv
    weights = np.random.randn(layers[1],layers[0])
    dataFile = open(File)
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
    dataFile = open(File)
    lines = dataFile.readlines()
    dataFile.close()
    counter = 0
    for neuron in range(layers[1]):
        for w in range(layers[0]):
            weights[neuron][w] = lines[counter]
            counter += 1
    return weights


def loadECG():
    '''
    function to load EEG data from their respective text files
    Each input in every list is a list in itself with the first list value being the 
    eeg signal in discrete time, and the second value of the list being the numeric label.
    ie. [[0,0,0,0,.4,.34,.23, .... ,0,0],[1]] 
    '''
    import csv
    import random
    test_data = []
    test_labels = []
    train_data = []
    train_labels = []
    print('##############')
    print('Loading Dataset')
    print('##############')
    with open('DataSets/Arrhythmia_Train.txt') as csv_file:
        lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in lines:
            train_data.append(list(row[0:-2]))
            train_labels.append(list(row[-2:-1]))
        csv_file.close()
    with open('DataSets/Arrhythmia_Test.txt') as csv_file:
        lines = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in lines:
            test_data.append(list(row[0:-2]))
            test_labels.append(list(row[-2:-1]))
        csv_file.close()

    test_data = list(zip(test_data,test_labels))
    train_data = list(zip(train_data,train_labels))
    return train_data,test_data


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
    print('##############')
    print('Loading Dataset')
    print('##############')
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
