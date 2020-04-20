# Paramater Definitions

max_epochs = 100
no = .01
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 100
#batch_size = 50                # uncomment for batch training
trainBool = True
layers = [1000000,100]
pathExt = "SavedWeights/FromBayley/1000k/savedWeights/"
SimParm = pathExt+"BestParam.txt"
Metrics = pathExt+"metrics.p"
MapPickle = pathExt+"map.p"
BMUPickle = pathExt+"BMUWeights.p"
Weights = pathExt+"Weights.txt"
OutputMap = pathExt+"Map.tiff"
OutputW =  pathExt+"Weights.tiff"
OutputMet = pathExt+"Metrics.tiff"
TestNN = pathExt+"testNN.txt"
OutNN = pathExt+"outNN.txt"
