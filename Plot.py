import funcs
from paramsE import *

weight = funcs.threshWeights('ECG/Weights.txt',layers,0.4)
#weight = funcs.loadWeights(OutputW,layers) # uncomment to plot true weight map
funcs.plotMetrics(max_epochs,Metrics,OutputMet,tau,tauN,no,sigmaP)
funcs.graphHeatmap(MapPickle,OutputMap)
#funcs.weightPlot(weight,OutputW)
