import NetworkClasses as NC
import random
import math
import numpy as np

random.seed(10)
def convertCharToInputVector(char):
    input_vector = []
    index = ord(char) - 65
    for x in range(36):
        if(x == index):
            input_vector.append(1)
        else:
            input_vector.append(0)
    return input_vector

def convertOutputLayerToChar(output_layer):
     final_output_vector = []
     for x in range(len(output_layer)):
        final_output_vector.append((output_layer[x].output))
        #print("outputlayer values " ,output_layerW[x].output)
     choice = max(final_output_vector)
     choice_index = final_output_vector.index((choice))
     char_output = chr(choice_index + 65)
     return  char_output

def create_weights(amount):
    weights = []
    for x in range(amount): # wa, wi
        weights.append(NC.Weight( random.uniform(0.2,1), random.uniform(0,1)))
    return weights

def PrintAE(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
                print("Layer:",x , "Neuron: ",y, "AE: ",topology[x][y].AE)

def PrintSkippedNeurons(skippedNeurons):
    data = np.array(skippedNeurons)
    data_t = data.T
    for x in range(len(data_t)):
        total_skipped = sum(data_t[x])
        print("Layer", x, " total skipped:" ,total_skipped, "average skipped per itr: ", total_skipped / len(data_t[x]))




def PrintWAMultiplier(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            print("Layer:",x , "Neuron: ",y, "WA Multiplier: ",topology[x][y].wa_multiplier)

def PrintR(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
                print("Layer:",x , "Neuron: ",y, "R: ",topology[x][y].R)

def CreateTopFromIndexes(indexes, p_topology):
    topology = []

    for x in reversed(range(len(indexes))):
        if x == len(p_topology) -1:
            topology.append(p_topology[len(p_topology) -1]) #outputLayer
            print("addiung output layer")
        else:
            topology.append([])
            for y in range(len(indexes[x])):
                #just add the weights that
                print("neurons, can be inactive")

    return topology

def removeWeightsFromLayer(layer, weightIndexesToBeRemoved):
    for x in range(len(layer)):
        for y in range(len(weightIndexesToBeRemoved)):
            layer[x].weights.pop(weightIndexesToBeRemoved[y])

    return layer

def clearSOArrays(topology):
    for x in range(len(topology) -1 ):
        for y in range(len(topology[x])):
            topology[x][y].SO.clear()

def PrintBias(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            print("Layer ", x , "Neuron" , y, "Bias" , topology[x][y].bias)

def getLayerOutputs(layer):
    outputVector = []
    for x in range(len(layer)):
        outputVector.append(layer[x].output)
    return outputVector

def getLayerBiases(layer):
    outputVector = []
    for x in range(len(layer)):
        outputVector.append(layer[x].bias)
    return outputVector

def clearTopologyArrays(topology_errors, topology_outputs):
    for x in range(len(topology_errors)):
        for y in range(len(topology_errors[x])):
            topology_errors[x][y] = 0

    for x in range(len(topology_outputs)):
        for y in range(len(topology_outputs[x])):
            topology_outputs[x][y] = 0


def addTopologyOutputs(topology_outputs, topology):
    for x in range(len(topology_outputs)):
        for y in range(len(topology_outputs[x])):
            topology_outputs[x][y] += topology[x][y].output
    return topology_outputs

def addTopologyErrors(topology_errors,topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            topology_errors[x][y] += topology[x][y].error
    return topology_errors

def getLayerErrors(layer, absValues):
    outputVector = []
    for x in range(len(layer)):
        if absValues == True:
            outputVector.append(abs(layer[x].error))
        else:
            outputVector.append(layer[x].error)
    return  outputVector

def PrintActivations(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            print("Layer:",x , "Neuron: ",y, "Activation: ",topology[x][y].activation)



def PrintActivationsTopology(topology):
    results = []
    for x in range(len(topology)):
        results.append([])
        topString = ""
        for y in range(len(topology[x])):
            if topology[x][y].activation >= 1:
                topString += "1"
            else:
                topString += "0"
        results[x].append(topString)
    print(results)

def getGlobalError(numRows, numTargetClass, predicted, target):

 #  print(numRows, numTargetClass, predicted, target)
    error = 0
    for i in range(numRows):
        for j in range(numTargetClass):
            error += target[i][j] * math.log(predicted[i][j])
    globalError = ((-1.0/numRows) * error)
    return globalError

def create_layer(layer_size, neuron_weight_amount, layerIndex):
    layer = []
    print("creating layer," ,layer_size , neuron_weight_amount, layerIndex)
    for x in range(layer_size):

        ID ="L" +str(layerIndex) + "N"+str(x)
        layer.append(NC.Neuron(neuron_weight_amount,ID ))
    return layer

def create_topology_dictionary(topology):
    dict = {}
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            dict[topology[x][y].ID] = topology[x][y]
    return dict






def calculateAverageLayerValues(value_array):
    #this one does not work
    errors = []
    print("VA", value_array)
    for x in range(len(value_array)):
        error = 0
        for y in range(len(value_array[x])):
            error += value_array[x][y]
            if y == len(value_array[x]) -1:
                error = error / len(value_array[x])
                errors.append(error)
    return errors


def avarageOutTopologyValues(topology_array,batchSize):
    for x in range(len(topology_array)):
        for y in range(len(topology_array[x])):
            topology_array[x][y] =  topology_array[x][y] / batchSize

    return topology_array

def create_topology(topology_size):
    topology = []
    for x in range(len(topology_size)):
        if x == 0:
            input_layer = create_layer(topology_size[x],topology_size[x +1],x)
            topology.append(input_layer)
        elif x == len(topology_size) - 1:
            output_layer = create_layer(topology_size[x], 0,x)
            topology.append(output_layer)
        else:
            print("creating hidden layer: size" ,topology_size[x]," weights" , topology_size[x+1])
            HL = create_layer(topology_size[x],topology_size[x+1],x)
            topology.append(HL)
    print("Created Topology")
    return topology

def PrintOutputs(topology):
     for x in range(len(topology)):
        for y in range(len(topology[x])):
            print("Layer:",x , "Neuron: ",y, "Value: ",topology[x][y].output)

def PrintWeights(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            for z in range(len(topology[x][y].weights)):
                print("Layer:",x , "Neuron: ",y, "WI: " , z , "Value: ",topology[x][y].weights[z].w_i)

def PrintWAWeights(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            for z in range(len(topology[x][y].weights)):
                print("Layer:",x , "Neuron: ",y, "WA: " , z , "Value: ",topology[x][y].weights[z].w_a)

def PrintErrors(topology):
    for x in range(len(topology)):
        for y in range(len(topology[x])):
            print("Layer:",x , "Neuron: ",y, "Error: ",topology[x][y].error)


def setInputLayerValues(inputVector, topology):
    for x in range(len(topology[0])):
        topology[0][x].output = inputVector[x]
        print(topology[0][x].output)

def isCorrectlyClassified(target, output):
    rounded = []
    for x in range(len(output)):
        rounded.append(round(output[x]))

    if rounded == target:
        return 1
    else:
        return 0

