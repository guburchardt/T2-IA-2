import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        if weights is None:
            self.weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
            self.weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        else:
            self.weights1 = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
            self.weights2 = weights[input_size * hidden_size:].reshape(hidden_size, output_size)

    def forward(self, inputs):
        hidden = np.dot(inputs, self.weights1)
        hidden_activation = np.tanh(hidden)
        output = np.dot(hidden_activation, self.weights2)

        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        return probabilities
