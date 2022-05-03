import random
import HelperFunctions as HF

random.seed(9)
class Weight:
    def __init__(self, w_a, w_i):
        self.w_a = w_a
        self.w_i = w_i

class Neuron:
    def __init__(self, weight_amount , ID):
        self.error = 0.0
        self.bias = random.uniform(-1,1)
        self.activation = 0.0
        self.weights = HF.create_weights(weight_amount)
        self.output = 0.0
        self.R = random.uniform(0,1)
        self.F = 0.85
        self.AE = 0.001
        self.ID = ID
        self.wa_multiplier = 0.0001




