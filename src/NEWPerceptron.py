from math import exp
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from src.NetworkEnums import ActivationFunctions    # noqa: E402


class Perceptron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.inputs = []
        self.activation = activation

    def getSum(self):
        _sum = 0
        for weight, input in zip(self.weights, self.inputs):
            _sum += weight * input
        return _sum + self.bias

    def SigmoidUnipolar(self):
        return 1 / (1 + exp(-self.getSum()))

    def SigmoidBipolar(self):
        return - 1 + 2 / (1 + exp(-self.getSum()))

    def RELU(self):
        return max(0, self.getSum())

    def Linear(self):
        return self.getSum()

    def DerSigUnipolar(self):
        result = self.SigmoidUnipolar()
        return result * (1 - result)

    def DerSigBipolar(self):
        result = self.SigmoidBipolar()
        return 0.5 * (1 - result ** 2)

    def DerRELU(self):
        return 1 if self.getSum() > 0 else 0


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons
        self.activation = ActivationFunctions.SIGMOID_UNIPOLAR
        self.perceptrons = []

    def createPerceptrons(self, weights, bias):
        for i in range(self.neurons):
            self.perceptrons.append(Perceptron(weights[i], bias[i]))

    def getOutputs(self):
        outputs = []
        for perceptron in self.perceptrons:
            outputs.append(perceptron.SigmoidUnipolar())
        return outputs

    def getDerivatives(self):
        derivatives = []
        for perceptron in self.perceptrons:
            derivatives.append(perceptron.DerSigUnipolar())
        return derivatives