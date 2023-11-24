import numpy as np


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, momentum=0.9, min_error=1e-5):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_error = min_error
        # self.weights = [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.weights = [np.array([[0.46224844, 0.31102436],
       [0.02655818, 0.57753249]]), np.array([[0.93985102, 0.92043428],
       [0.23729138, 0.89560753]]), np.array([[0.12755261],
       [0.82588899]])]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        self.activations = [self.sigmoid for _ in range(len(layers)-2)]
        self.activations.append(self.sigmoid_output)

        # Inicjalizacja prędkości (momentum) dla wag
        self.velocity_weights = [np.zeros((layers[i], layers[i+1])) for i in range(len(layers)-1)]
        self.velocity_biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output(self, x):
        # Alternatywna sigmoida dla warstwy wyjściowej
        return 2 / (1 + np.exp(-x)) - 1

    def forward(self, inputs):
        self.outputs = [inputs]
        for i in range(len(self.layers)-1):
            inputs = self.activations[i](np.dot(inputs, self.weights[i]) + self.biases[i])
            self.outputs.append(inputs)
        return inputs

    def train(self, inputs, targets, epochs=10000):
        for epoch in range(epochs):
            # Przemieszaj dane przed każdą epoką
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            shuffled_inputs = inputs[indices]
            shuffled_targets = targets[indices]

            total_error = 0
            for i in range(len(shuffled_inputs)):
                input_data = np.array([shuffled_inputs[i]])
                target_data = np.array([shuffled_targets[i]])
                total_error += self.train_single(input_data, target_data)
            # for i in range(4):
            #     input_data = np.array([inputs[i]])
            #     target_data = np.array([targets[i]])
            #     total_error += self.train_single(input_data, target_data)
            average_error = total_error / len(shuffled_inputs)
            if average_error < self.min_error:
                print(f"Trening zakończony po {epoch+1} epokach. Średni błąd: {average_error}")
                break

    def train_single(self, input_data, target_data):
        # Forward pass
        # set_data = np.array([[1, 0]])
        # new_target = np.array([[1]])
        self.forward(input_data)
        # self.forward(set_data)
        # print(input_data)
        # print(self.outputs)

        # Backpropagation
        errors = [target_data - self.outputs[-1]]
        # errors = [new_target - self.outputs[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative_output(self.outputs[-1])]

        for i in range(len(self.layers) - 2, 0, -1):
            errors.insert(0, deltas[0].dot(self.weights[i].T))
            deltas.insert(0, errors[0] * self.sigmoid_derivative(self.outputs[i]))
        # print(errors)
        # Update weights and biases with momentum
        total_error = np.sum(errors[-1]**2)  # Calculate total error for monitoring

        for i in range(len(self.layers)-1):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * self.outputs[i].T.dot(deltas[i])
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

            self.weights[i] += self.velocity_weights[i]
            self.biases[i] += self.velocity_biases[i]

        return total_error

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def sigmoid_derivative_output(self, x):
        return 0.5 * (1 - x**2)

# Przykład użycia dla funkcji XOR z learning rate = 0.1, momentum = 0.9 i min_error = 1e-5:
nn = NeuralNetwork([2, 2, 2, 1], learning_rate=0.1, momentum=0.9, min_error=1e-5)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

nn.train(inputs, targets)


nn.train(inputs, targets)

for i in range(len(inputs)):
    output = nn.forward(inputs[i])
    print(f"Wejścia: {inputs[i]}, Wyjście: {output}")