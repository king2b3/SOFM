def graphHeatmap(Input,Output):
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


def loadWeights(File,layers):
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
    import numpy as np
    return (np.transpose(weights[num].reshape(size,size)))

def weightPlot(weights,Output):
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
    return val[1] 

def plotMetrics(max_epochs,Metrics,Output):
    import matplotlib.pyplot as plt
    import pickle as pkl
    plt.figure()
    epochs = range(max_epochs)
    path1 = 'SavedWeights/'+Metrics
    path2 = 'SavedWeights/'+Output
    metrics = pkl.load( open(path1, "rb" ) )
    plt.plot(epochs, metrics, 'ro')
    plt.xlabel('Epohcs')
    plt.ylabel('Average distance')
    plt.title('Average distance between 1st and 2nd winning neuron over each epoch')
    plt.ylim([0,4])
    #plt.show()
    plt.savefig(path2)

def plotNeuronMap(weights,Output):
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