import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoidDerivative(output):
    return output * (1.0 - output)

def reverseSigmoidDerivate(output):
    return output / (output + 1.0)

def relu(x):
    return max(0.0, x)

def reluDerivate(x):
    y = (x > 0) * 1
    return y

def distance(a, b):
    d = math.sqrt((a - b)**2)
    return d

def clamp(num, min_value, max_value):
    num = max(min(num, max_value), min_value)
    return num

def moveTowards(input, target, speed):
    if input > target:
        return input - speed
    else:
        return input + speed

def reverseSigmoid(y):
    return math.log(y/(1-y))

def sigmoid(x):
    try:
        sig = 1 / (1 + math.exp(-x))
        return sig
    except OverflowError:
        return 0.9999999

#https://www.geeksforgeeks.org/how-to-make-a-bell-curve-in-python/
# probability distribution function
def pdf(x ,r ,f):
    mean = np.mean(r) # THis should be Resoance (R ) this decides where the peak of the bell curve
    std = f
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out

def curveVisual():
    curve_test = np.arange(0, 1.0, 0.05)
    curve_values = []
    f = 0.4
    mean = 0.3
    for index in range(len(curve_test)):
        value = pdf(curve_test[index], mean, f)
        max_value = pdf( mean, mean, f)  #insertMean to get maxValue
        curve_values.append(value / max_value)

    plt.style.use('seaborn')
    plt.figure(figsize = (6, 6))
    plt.plot(curve_test, curve_values, color = 'black',
             linestyle = 'dashed')
    plt.scatter( curve_test, curve_values, marker = 'o', s = 25, color = 'red')
    plt.show()


def unrollOError(error, output):
    #reverse of output-target * f'(output)
    x_neg = error / output - output
    return abs(x_neg)

def unrollHError(error, output):

    #weights * error * f'(output)
    #E = Wn * Em
    #E = E * f'(output)
    return 0.5
