# Paramater Definitions

max_epochs = 15
no = 1
tau = max_epochs/2
tauN = max_epochs/5
sigmaP = 100
#batch_size = 50                # uncomment for batch training
trainBool = True
layers = [187,100]
SimParm = "ECG/BestParam.txt"
Metrics = "ECG/metrics.p"
MapPickle = "ECG/map.p"
Weights = "ECG/Weights.txt"
OutputMap = "ECG/Map.tiff"
OutputW =  "ECG/Weights.tiff"
OutputMet = "ECG/Metrics.tiff"
#OutputN1 = "Neruon 83.tiff"    # plot specific neurons, if you want
#OutputN2 = "Neruon 2.tiff"