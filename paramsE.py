# Paramater Definitions

max_epochs = 50
no = 1
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 100
#batch_size = 50                # uncomment for batch training
trainBool = True
layers = [187,100]
pathExt = "SavedWeights/ECG/"
SimParm = "SavedWeights/ECG/BestParam.txt"
Metrics = "SavedWeights/ECG/metrics.p"
MapPickle = "SavedWeights/ECG/map.p"
BMUPickle = pathExt+"BMUWeights.p"
Weights = "SavedWeights/ECG/Weights.txt"
OutputMap = "SavedWeights/ECG/Map.tiff"
OutputW =  "SavedWeights/ECG/Weights.tiff"
OutputMet = "SavedWeights/ECG/Metrics.tiff"
#OutputN1 = "Neruon 83.tiff"    # plot specific neurons, if you want
#OutputN2 = "Neruon 2.tiff"