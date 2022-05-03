import math
import random

import HelperFunctions as HF
import MathLib
import NetworkClasses as NC
import numpy as np
learning_rate = 0.05
wa_learning_rate = 0.01
clampValue = 10
inDistrubtionMultiplier = 0.01
outDistrubtrionMultipler = 2

def backPropagate(target, topology, activeIndexes, topDict):
    for i in reversed(range(len(topology))):
        if i != len(topology) - 1 :
            if i == 0:
                break
            for j in range(len(activeIndexes[i-1])):
                error = 0
                key = "L" + str(i) + "N" + str(activeIndexes[i-1][j])
                if i == len(topology) -2:
                    for x in range(len(topology[i + 1])):
                        error += (topDict[key].weights[x].w_i * topology[i+1][x].error)
                    topDict[key].error = error * MathLib.sigmoidDerivative(topDict[key].output)
                else:
                    for x in range(len(activeIndexes[i])):
                        key2 = "L" + str(i+ 1) + "N" + str(activeIndexes[i][x])
                        error += (topDict[key].weights[x].w_i * topDict[key2].error)
                    topDict[key].error = error * MathLib.sigmoidDerivative(topDict[key].output)




            layerErrors = HF.getLayerErrors(topology[i],True)
            if len(layerErrors) == 0:
                continue
            mean_layer_error = (sum(layerErrors)) / len(layerErrors)
            if mean_layer_error == 0:
                continue

            mean_distance_to_mean_error = 0
            max_distance = 0
            for x in range(len(layerErrors)):
                d = abs(mean_layer_error-layerErrors[x])
                if d > max_distance:
                    max_distance = d
                mean_distance_to_mean_error += d
            mean_distance_to_mean_error = mean_distance_to_mean_error / len(layerErrors)
            #ae_multiplier = MMD / mean_layer_error


            for j in range(len(topology[i])):
                #if our |E| is smaller than median E, we want the neuron to be boosted
                #else it should get a negative signal
                E1 = abs(topology[i][j].error) / mean_layer_error
                #if E1 < 1, we our error is smaller than the average error
                #if it is, we want the neurons wa to be boosted = positive AE
                E3 = 1 - E1
                #E3 will be negative if want to reduce WAs = negatvie AE
                #therefore, E3 * the distance multiplier, E3 retains some relation to its previous state... 0.0001 -> 20... weird.
                Distance = abs(mean_layer_error - topology[i][j].error)
                E2 = Distance / mean_distance_to_mean_error
                #if E2 < 1, it means the distance to the mean error is below average, meaning that its NOT an outlier
                E4 = 0
                if E2 < 1:
                    E4 = inDistrubtionMultiplier* (1-E2)
                else:
                    E4 = outDistrubtrionMultipler * (max_distance -E2) #this normalizes the value (kinda)
                #E4 Need to become the actual multiplier that we use to multiply E3 with
                E5 = E3 * E4
                topology[i][j].AE = E5
          #  print("meanError", mean_distance_to_mean_error ,"Distance List", sorted(Distance_List))


        else: #output layer, can stay the same
            for j in range(len(topology[i])): #neuron
                delta = topology[i][j].output - target[j]
                topology[i][j].error = delta #* MathLib.sigmoidDerivative(topology[i][j].output)

def calculateRStep(neuron,wa, connectedNeuron):
    contribution = neuron.wa_multiplier * wa #this is how much we GAVE the neuron
    AE = connectedNeuron.AE
    if AE == 0: #First iteration failsafe
        return neuron.R
    if contribution >= 0:
        value = AE / contribution #if AE is Pos and C is pos, we get pos
        return MathLib.moveTowards(neuron.R, neuron.output, value * wa_learning_rate)
    else:
        value = AE / contribution #if AE is Pos and C is pos, we get pos
        return MathLib.moveTowards(neuron.R, neuron.output, -value * wa_learning_rate)


def adjustWeights(topology, _inputs, activeIndexes, topDict):
    for i in range(len(topology)):  # 2,1, #layer
        if i == len(topology) - 1:
            continue
        if i == 0:
            inputs = _inputs
        else:
            inputs = HF.getLayerOutputs(topology[i-1])

        if i == 0: #input layer
             for y in range(len(topology[i])):
                score = 0
                for j in range(len(activeIndexes[0])):
                    key = "L" + str(i+1) + "N" + str(activeIndexes[0][j])

                    score += topDict[key].AE / ( topology[i][y].weights[j].w_a  * topology[i][y].wa_multiplier)
                    topology[i][y].weights[j].w_a += topDict[key].AE * wa_learning_rate * topology[i][y].wa_multiplier
                    topology[i][y].weights[j].w_a = MathLib.clamp(topology[i][y].weights[j].w_a,-clampValue,clampValue)

                    topology[i][y].weights[j].w_i += -learning_rate * topDict[key].error * inputs[y]
                topology[i][y].bias += topology[i][y].error * -learning_rate

                topology[i][y].R = MathLib.moveTowards(topology[i][y].R, topology[i][y].output , math.tanh(score) * wa_learning_rate)

                #topology[i][y].R = MathLib.clamp(topology[i][y].R , 0,1)


        elif i < len(topology) -2:
            for y in range(len(activeIndexes[i-1])): #this
                score = 0
                key = "L" + str(i) + "N" + str(activeIndexes[i-1][y])

                for j in range(len(activeIndexes[i])):
                    key2 = "L" + str(i +1) + "N" + str(activeIndexes[i][j])
                    #topDict[key].R = calculateRStep(topology[i][y],  topDict[key].weights[activeIndexes[i][j]].w_a ,  topDict[key2])

                    score += topDict[key2].AE / (  topDict[key].weights[activeIndexes[i][j]].w_a  * topDict[key].wa_multiplier)
                    topDict[key].weights[activeIndexes[i][j]].w_a += topDict[key2].AE * wa_learning_rate * topDict[key].wa_multiplier
                    topDict[key].weights[activeIndexes[i][j]].w_a = MathLib.clamp(topDict[key].weights[activeIndexes[i][j]].w_a, -clampValue,clampValue)

                    for k in range(len(inputs)):
                        topDict[key].weights[j].w_i += -learning_rate * topDict[key2].error * inputs[k]

                topDict[key].bias += topDict[key].error * -learning_rate
                topDict[key].R = MathLib.moveTowards( topDict[key].R,  topDict[key].output,  math.tanh(score) * wa_learning_rate)
        #        topDict[key].R = MathLib.clamp( topDict[key].R , 0,1)

        else: #should be output layer? , layer index 2 in 4 layer setup
            for y in range(len(activeIndexes[i-1])): #NEUORNS
                key = "L" + str(i) + "N" + str(activeIndexes[i-1][y])
                for j in range(len(topology[len(topology)-1])): #WEIGHTS
                    for k in range(len(activeIndexes[i-2])):  #THESE ARE INPUTS
                        topDict[key].weights[j].w_i += -learning_rate * topology[i+1][j].error * topDict[key].output
                topDict[key].bias += topDict[key].error * -learning_rate
