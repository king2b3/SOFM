# Bayley King
# Python 3.7.3
# Ploting function library for SOFM network

import funcs

def graphHeatmap(Input,Output):
    '''
    Function to graph the final heatmap for the best input for each output neuron
    '''
    import matplotlib.pyplot as plt
    import pickle as pkl
    plt.figure()
    h = []
    output = pkl.load( open(Input, "rb" ) )
    for x in range(0, len(output), 10):  
        h.append(output[x:x + 10])
    import seaborn as sns; sns.set()
    ax = sns.heatmap(h, annot=True,xticklabels=False,yticklabels=False,cbar=False)
    #plt.show()
    plt.savefig(Output)


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
            axarr[row,col].imshow(lookAtTheseBoys(numfella,weights,100),cmap=plt.get_cmap('gray_r'))
            #axarr[row,col].grid('on', linestyle='--')
            axarr[row,col].set_xticklabels([])
            axarr[row,col].set_yticklabels([])
            axarr[row,col].set_aspect('equal')
            axarr[row,col].axis('off')
            numfella += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(Output)
    #plt.show()


def plotMetrics(max_epochs,Metrics,Output,tau,tauN,no,sigmaP):
    '''
    plots the %decrease in the neighborhood rate and the learning rate with the 
    average distance between the 1st and 2nd BMU against epochs
    '''
    S = []
    L = []

    for epochs in range(max_epochs):
        S.append(funcs.sigma(epochs,tauN,sigmaP)/sigmaP)
        L.append(funcs.decay_LR(epochs,tau,no)/no)


    import matplotlib.pyplot as plt
    import pickle as pkl

    #plt.figure()
    fig, ax1 = plt.subplots()
    epochs = range(max_epochs)
    metrics = pkl.load(open(Metrics, "rb" ))
    print(len(epochs))
    print(len(metrics))

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
    plt.savefig(Output)


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
    plt.savefig(Output)
    #plt.show()