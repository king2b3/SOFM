# Bayley King
# Python 3.7.3
# Metric function library for SOFM network 

import funcs

def neighborhoodPreservation(Input,test):
    '''
    this function will output the neighborhood preservation factor of the trained map
    ie. the number of labels in the output map will be compared to the number of labels
        in the testing space. if trained properly the two should have similar distrobutions
    '''
    import pickle as pkl
    import numpy as np
    outMap = []
    path1 = 'SavedWeights/'+Input
    outMap = pkl.load( open(path1, "rb" ) )
    testlabels = []
    testlabels_counter = []
    test.sort(key=lambda x: x[1])
    for t in test:
        if t[1] not in testlabels:
            testlabels.append(t[1])
            testlabels_counter.append(0)
        loc = testlabels.index(t[1])
        testlabels_counter[loc] += 1
    testlabels = [elem[0] for elem in testlabels]
    test_trust = list(zip(testlabels,testlabels_counter))

    outMap.sort()
    labels = []
    for l in testlabels:
        labels.append(l)
    labels_counter = list(np.zeros(len(labels)))
    for m in outMap:
        loc = labels.index(m)
        labels_counter[loc] += 1
    map_trust = list(zip(labels,labels_counter))

    score_map = []
    for m in map_trust:
        score_map.append(float(m[1]/len(outMap)))
    score_test = []
    for t in test_trust:
        score_test.append(float(t[1]/len(test)))

    score = 0.0
    for elem in range(len(score_map)):
        score += abs((score_map[elem] - score_test[elem])/score_test[elem])

    score_test = [ '%.2f' % elem for elem in score_test ]
    score_map = [ '%.2f' % elem for elem in score_map ]
    testScore = list(zip(testlabels,score_test))
    mapScore = list(zip(labels,score_map))
    print('ROUNDED Scores')
    print(mapScore)
    print(testScore)
    print(score)


def genTestTrustworthiness(test,outFilename,k=4):
    '''
    This function will generate a text file that contains the k nearnest neighbors for each input in the
        testing set. The saved values are the indicies for the neighbors for the respective testing point 
    '''
    import numpy as np
    import csv
    from csv import writer
    test = np.array(test)
    with open(outFilename, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        print('\nFinding Neighbors...\n')
        for a in test:
            c = abs(a[0]-test)
            d = sum(np.transpose(c))
            neighbors = []
            for i in range(k):
                e = np.argmin(d)
                neighbors.append(e)
                d[e] = 9999999999999
            csv_writer.writerow(neighbors[1:])


def outputBMU(BMUFilename,TestFilename):
    '''
    outputBMU will print the total accuaracy of the trustworthiness of the network
    Everytime a the current neighbor of an input is also a neighbor of the same index
        in the output set, a hit is recorded, else a miss is recorded.
    The accuracy is simply the hits/total points.
    '''
    print('Loading BMU Weights KNNs......')
    dataFile = open(BMUFilename)
    BMUlines = dataFile.readlines()
    dataFile.close()
    print('Loading Test KNNs......')
    dataFile = open(TestFilename)
    Testlines = dataFile.readlines()
    dataFile.close()
    
    hit = miss = 0
    for (B1,T1) in zip(BMUlines,Testlines):
        for B in B1:
            if B in T1:
                #print('hit!',B,T)
                hit += 1
            else:
                #print('miss!',B,T)
                miss += 1

    print('Total accuracy:',hit/(hit+miss))


def main():
    print('ECG')
    train,test = funcs.loadECG()
    pathExt = "SavedWeights/ECG/"
    TestNN = pathExt+"testNN.txt"
    OutNN = pathExt+"outNN.txt"
    BMUPickle = pathExt+"BMUWeights.p"
    BMU = pkl.load(open(BMUPickle, "rb" ))
    genTestTrustworthiness(BMU,OutNN)
    genTestTrustworthiness(test,TestNN)
    outputBMU(OutNN,TestNN)

if if __name__ == "__main__":
    main()