import numpy as np

np.random.seed(0)
# Converting to class based
# X is input data to Neural Network
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


# 2 Hidden Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # n_inputs: size of the input coming in (aka size of single sample eg: [1, 2, 3, 2] = 4)
        # n_neurons: how many neurons do we want to have

        # randn generates an array of random numbers of size row x col
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Take the dot product of the inputs and the weights
        # Note the inputs to one layer can be the output from another
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)  # X has 4 parts
layer2 = Layer_Dense(5, 2)  # Note 1st layer outputs 5 neurons, so layer 2 must input 5, however the output can be anything

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)