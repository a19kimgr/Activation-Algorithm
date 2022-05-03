import ActivationPropagation as AP
import HelperFunctions as HF
import MathLib
def calculateNeuronOutput(PreviousLayer, currentNeuron, current_neuron_y):
    _output = 0
    for x in range(len(PreviousLayer)):
        _output += PreviousLayer[x].output * PreviousLayer[x].weights[current_neuron_y].w_i
    value = MathLib.sigmoid(_output + currentNeuron.bias)
    return value


def calculateNeuronOutputIndex(PreviousActiveIndexes, currentNeuron, current_neuron_y, topDict, layerIndex):
    _output = 0
    for x in range(len(PreviousActiveIndexes)):
        key = "L" +str(layerIndex-1) + "N"+str(PreviousActiveIndexes[x])
      #  print(len(topDict[key].weights), current_neuron_y)
        _output += topDict[key].output * topDict[key].weights[current_neuron_y].w_i
    value = MathLib.sigmoid(_output + currentNeuron.bias)
    return value





def propagateTopology(topology, input, topDict):
    dynamicTop = []
    for x in range(len(topology)):
        #print(dynamicTop)
        if x == 0: #input neurons
            setInputNeurons(input, topology)
            dynamicTop.append(AP.propagateInputLayer(topology)) #Input WA muiltipler set
        else:
            if x != len(topology)-1: #skit calculatiing acvtviaton for outputlayer
                propagateActivationIndexOutput(x, dynamicTop,topDict, topology) #as for now this calcualtes output for the layers,
                if x != len(topology) -2:
                    dynamicTop.append(AP.propagateHiddenLayer(dynamicTop[len(dynamicTop)-1], x, topology[x+1],topDict))

            else:
                for y in range(len(topology[x])):
                    topology[x][y].output = calculateNeuronOutputIndex(dynamicTop[x -2],topology[x][y], y,topDict, x)

    return dynamicTop


def propagateActivationIndexOutput(layerIndex,activeIndexes,topDict, topology):
    #calculate the new output values

    for x in range(len(activeIndexes[layerIndex -1])):
        key = "L"+str(layerIndex) + "N"+str(x)
        if layerIndex == 1: #we using input layer
            topDict[key].output = calculateNeuronOutput(topology[0],  topDict[key], activeIndexes[layerIndex -1][x])

            #SET this neurons WA multipler here
        else:
            #the previous layer that was active, this has to be done through getting the last activeIndexes
            topDict[key].output = calculateNeuronOutputIndex(activeIndexes[layerIndex -2],topDict[key], x,topDict, layerIndex)
             #SET this neurons WA multipler here
            #print("HL Output: ", topDict[key].output)
        distance = MathLib.distance( topDict[key].output,  topDict[key].R)
        m1 =  topDict[key].F - distance
        wa_multiplier = MathLib.clamp(m1,-1,1)
        topDict[key].wa_multiplier = wa_multiplier

def setInputNeurons(input, topology):
    inputNeurons = topology[0]
    for x in range(len(inputNeurons)):
        inputNeurons[x].output = input[x]

        #Should be able to set WA multiplier here
        distance = MathLib.distance(inputNeurons[x].output, inputNeurons[x].R)
        m1 = inputNeurons[x].F - distance
        wa_multiplier = MathLib.clamp(m1,-1,1)
        inputNeurons[x].wa_multiplier = wa_multiplier



