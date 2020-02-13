def graphHeatmap():
    import matplotlib.pyplot as plt
    import pickle as pkl
    h = []
    output = pkl.load( open( "SavedWeights/map.p", "rb" ) )
    for x in range(0, len(output), 10):  
        h.append(output[x:x + 10])
    import seaborn as sns; sns.set()
    ax = sns.heatmap(h, annot=True,xticklabels=False,yticklabels=False,cbar=False)
    #plt.show()
    plt.savefig("SavedWeights/Map.jpeg")


def loadWeights():
    import numpy as np
    import csv
    layers = [784,100]
    weights = np.random.randn(layers[1],layers[0])
    with open('SavedWeights/Weights.txt', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for neuron in range(len(weights)):
            for w in range(len(weights[0])):
                csv_writer.writerow([weights[neuron][w]])
    return weights


def lookAtTheseBoys(num,weights):
    return (weights[num].reshape(28,28))

def weightPlot(weights):
    import matplotlib.pyplot as plt
    numfella = 0
    f, axarr = plt.subplots(10,10) #,constrained_layout=True)#,gridspec_kw = {'wspace':0, 'hspace':0})
    for row in range(10):
        for col in range(10):
            axarr[row,col].imshow(lookAtTheseBoys(numfella,weights),cmap=plt.get_cmap('gray_r'))
            #axarr[row,col].grid('on', linestyle='--')
            axarr[row,col].set_xticklabels([])
            axarr[row,col].set_yticklabels([])
            axarr[row,col].set_aspect('equal')
            axarr[row,col].axis('off')
            numfella += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("SavedWeights/Weights.tiff")
    #plt.show()

w = loadWeights()
weightPlot(w)