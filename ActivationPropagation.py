import MathLib
import HelperFunctions as HF
def calculateNeuronActivation(PreviousLayer, Y_index, currentNeuron):
    value = 0
    for x in range(len(PreviousLayer)):
        value += PreviousLayer[x].weights[Y_index].w_a * PreviousLayer[x].wa_multiplier
    return value

def calculateNeuronActivaitonIndex(PrevActivationIndexes, Y_index, currentNeuron, topDict, layerIndex):
    value = 0
    for x in range(len(PrevActivationIndexes)):
        key = "L"+str(layerIndex) + "N"+ str(PrevActivationIndexes[x])
        value += topDict[key].weights[Y_index].w_a * topDict[key].wa_multiplier

    return value


def propagateInputLayer(topology):
    activeIndexes = []
    for x in range(len(topology[1])):
        topology[1][x].activation = calculateNeuronActivation(topology[0],x, topology[1][x])
        if topology[1][x].activation >= 1:
            activeIndexes.append(x)
    return activeIndexes

def propagateHiddenLayer(PrevActivationIndexes, layerIndex, nextLayer , topDict): #
    activeIndexes = []
    for x in range(len(nextLayer)):
        nextLayer[x].activation = calculateNeuronActivaitonIndex(PrevActivationIndexes,x, nextLayer[x],topDict,layerIndex)
        if nextLayer[x].activation >= 1:
            activeIndexes.append(x)
    return activeIndexes


