import random
import ForwardPass as FP
import BackPropagation as BP
import HelperFunctions as HF
import DataSet as DS
import ActivationPropagation as AP
import NetworkClasses as NC
import math
import numpy as np
import matplotlib.pyplot as plt
import MathLib

topology_size = [54,15,15,7]
topology = HF.create_topology(topology_size)
topology_dictionary = HF.create_topology_dictionary(topology)
ds = DS.DataSet()

epoch_amount = 200
batch_size = 7
nrOfClasses = topology_size[len(topology_size) - 1]
correctly_classified = 0
accuracyArr = []
globalErrorArr = []
outputMatrix = []
skippedNeurons = []
accuracy = 0


print(epoch_amount * ds.getBatchTrainAmount() * batch_size * 60)
#HF.PrintWeights(topology)
#HF.PrintWAWeights(topology)

for i in range(0, epoch_amount):
    outputVector = []
    for x in range(0, ds.getBatchTrainAmount()):
        for y in range(0, batch_size):
            activeIndexes = FP.propagateTopology(topology, ds.getBatchTrainInputRow(x, y), topology_dictionary)

            BP.backPropagate(ds.getTrainBatchTarget(x, y, nrOfClasses), topology, activeIndexes, topology_dictionary)
            BP.adjustWeights(topology, ds.getBatchTrainInputRow(x, y), activeIndexes, topology_dictionary)
            outputMatrix.append(HF.getLayerOutputs(topology[len(topology) - 1]))
            correctly_classified += HF.isCorrectlyClassified(ds.getTrainBatchTarget(x, y, nrOfClasses), HF.getLayerOutputs(topology[len(topology)-1]))
            skippedNeurons.append([ topology_size[1] - len(activeIndexes[0]) , topology_size[2] - len(activeIndexes[1])])

    if correctly_classified != 0:
        accuracy = correctly_classified/(ds.getBatchTrainAmount() * batch_size)
        correctly_classified = 0

    globalError = HF.getGlobalError(ds.getBatchTrainAmount() * batch_size, nrOfClasses, outputMatrix, ds.getAllTargets())
    print("epoch: ", i, "Accuracy" , round(accuracy * 100, 3 ) , "%", "  Global Error ", globalError , " Skipped Neurons " , skippedNeurons[len(skippedNeurons)-1])
    outputMatrix.clear()
    accuracyArr.append(accuracy)
    globalErrorArr.append(globalError)

HF.PrintActivations(topology)
#HF.PrintWeights(topology)
HF.PrintR(topology)
HF.PrintWAWeights(topology)
#HF.PrintSkippedNeurons(skippedNeurons)
#HF.PrintAE(topology)
#HF.PrintBias(topology)
##HF.PrintErrors(topology)

ds.createGraph(skippedNeurons, ["Layer1", "Layer2"],"Skipped Neurons",epoch_amount * batch_size * ds.getBatchTrainAmount())
ds.createGraph(accuracyArr, ["Accuracy"], "%", epoch_amount)
ds.createGraph(globalErrorArr, ["Global Error"],"Error",epoch_amount )

outputMatrix.clear()
accuracyArr.clear()
globalErrorArr.clear()
skippedNeurons.clear()

print("DONE WITH TRAINING, TEST SET  COMMENCING")
print("DONE WITH TRAINING, TEST SET  COMMENCING")
print("DONE WITH TRAINING, TEST SET  COMMENCING")

for x in range(0,ds.getTestAmount()): #1):   #
    for y in range(0, batch_size):
        activeIndexes = FP.propagateTopology(topology, ds.getTestInputRow(x, y), topology_dictionary)
        #HF.PrintActivationsTopology(topology)
        outputMatrix.append(HF.getLayerOutputs(topology[len(topology) - 1]))
        correctly_classified += HF.isCorrectlyClassified(ds.getTestTarget(x, y, nrOfClasses), HF.getLayerOutputs(topology[len(topology)-1]))
        skippedNeurons.append([topology_size[1] - len(activeIndexes[0]) , topology_size[2] - len(activeIndexes[1])])

if correctly_classified != 0:
    accuracy = correctly_classified/(ds.getTestAmount() * batch_size)
    correctly_classified = 0

outputMatrix.clear()
accuracyArr.append(accuracy)
print("Test accuracy:" ,accuracyArr[0] * 100 , "%")
HF.PrintSkippedNeurons(skippedNeurons)

ds.createGraph(skippedNeurons, ["Layer1", "Layer2"],"Skipped Neurons",  batch_size * ds.getTestAmount())

outputMatrix.clear()
print("Done with everything")
