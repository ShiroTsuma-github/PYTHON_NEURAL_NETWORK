import math
import random

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [[random.random() for _ in range(layers[i+1])] for i in range(len(layers)-1)]
        self.biases = [0 for _ in range(len(layers)-1)]
        self.activations = [self.sigmoid for _ in range(len(layers)-2)]
        self.activations.append(self.sigmoid_output)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_output(self, x):
        return 2 / (1 + math.exp(-x)) - 1

    def forward(self, inputs):
        self.outputs = [inputs]
        for i in range(len(self.layers)-1):
            inputs = [self.activations[i](sum(w * x for w, x in zip(row, inputs)) + b) for row, b in zip(self.weights[i], self.biases[i])]
            self.outputs.append(inputs)
        return inputs[0]

    def train(self, inputs, targets, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target_data = targets[i]
                self.train_single(input_data, target_data, learning_rate)

    def train_single(self, input_data, target_data, learning_rate):
        # Forward pass
        self.forward(input_data)

        # Backpropagation
        errors = [target_data[0] - self.outputs[-1][0]]
        deltas = [errors[-1] * self.sigmoid_derivative_output(self.outputs[-1][0])]

        for i in range(len(self.layers)-2, 0, -1):
            errors.insert(0, sum(d * w for d, w in zip(deltas[0], self.weights[i])))
            deltas.insert(0, [errors[0] * self.sigmoid_derivative(self.outputs[i][j]) for j in range(len(self.outputs[i]))])

        # Update weights and biases
        for i in range(len(self.layers)-1):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += self.outputs[i][j] * deltas[i][j] * learning_rate
            self.biases[i] += sum(deltas[i]) * learning_rate

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def sigmoid_derivative_output(self, x):
        return 0.5 * (1 - x**2)

# Przykład użycia dla funkcji XOR:
# Utwórz sieć neuronową z dwiema warstwami ukrytymi: 2 wejścia, 2 neurony w każdej warstwie ukrytej, 1 wyjście
nn = NeuralNetwork([2, 2, 2, 1])

# Ucz sieć na przykładzie XOR
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

nn.train(inputs, targets)

# Testuj sieć
for i in range(len(inputs)):
    output = nn.forward(inputs[i])
    print(f"Wejścia: {inputs[i]}, Wyjście: {output}")
