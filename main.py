import math
import random
from csv import reader
import sys, os

class Data:
    def __init__(self, filename, has_header, delimiter_type):
        self.filename = filename
        self.path  = os.path.join(sys.path[0], self.filename)
        self.delimiter_type = delimiter_type
        self.has_header = has_header
        self.dataset = None

    def read_data(self):
        lines = reader(open(self.path, "r"), delimiter = self.delimiter_type)
        self.dataset = list(lines)
        if self.has_header:
            self.dataset.pop(0)
        for record in self.dataset:
            for i in range(len(record)-1):
                record[i] = float(record[i].strip())
            record[-1] = int(record[-1].strip())


    def normalize_dataset(self):
        minmax = []
        for column in zip(*self.dataset):
            minmax.append([min(column), max(column)])
        for row in self.dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

    def prepare_data(self):
        self.read_data()
        self.normalize_dataset()

    
class Neuron:
    def __init__(self, weights_number):
        # Init wights with rand numbers between 0 and 1, last weight is bias
        self.weights = [random.uniform(0, 1) for _ in range(weights_number)]
        self.output = 0.0
        self.error = None

    def activate(self, inputs):
        result = 0
        result += self.weights[-1]
        for i, input in enumerate(inputs):
            result += input * self.weights[i]
        return self.activation_function(result)

    def activation_function(self, sum_result):
        return 1.0 / (1.0 + math.exp(-sum_result))

    def activation_function_derivative(self):
        return self.output * (1 - self.output)


class Layer:
    def __init__(self, name, neurons_number, inputs_number):
        self.name = name
        # List of neurons in layer. +1 weight for bias
        self.neurons = [Neuron(inputs_number + 1) for _ in range(neurons_number)]


class NeuralNetwork:
    def __init__(self, input_number, hidden_number, output_number):
        self.layers = [Layer("hidden", hidden_number, input_number), Layer("output", output_number, hidden_number)]
        self.input_number = input_number
        self.hidden_number = hidden_number
        self.output_number = output_number

    def forward_propagation(self, inputs):
        for layer in self.layers:
            next_layer_inputs = []
            for neuron in layer.neurons:
                neuron.output = neuron.activate(inputs)
                next_layer_inputs.append(neuron.output)
            inputs = next_layer_inputs
        return inputs

    def backward_propagation(self, expected_outputs):
        for i, layer in reversed(list(enumerate(self.layers))):
            errors = []
            if i != 1:  # if layer is not output layer
                for j in range(len(layer.neurons)):
                    error = 0.0
                    for neuron in self.layers[i + 1].neurons:
                        error += neuron.weights[j] * neuron.error
                    errors.append(error)
            else:  # layer is output layer (last layer, first iteration in forloop)
                for j, neuron in enumerate(layer.neurons):
                    errors.append(neuron.output - expected_outputs[j])
            for j, neuron in enumerate(layer.neurons):
                neuron.error = errors[j] * neuron.activation_function_derivative()

    def update_weights(self, data, learn_rate):
        for i, layer in enumerate(self.layers):
            inputs = data[:-1]
            if i != 0:  # layer is not first hidden
                inputs = [neuron.output for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron.weights[j] -= learn_rate * neuron.error * inputs[j]
                neuron.weights[-1] -= learn_rate * neuron.error

    def train(self, epochs, learn_rate, train_data):
        for epoch in range(epochs-5):
            error_sum = 0.0
            for data in train_data:
                outputs = self.forward_propagation(data)
                expected_outputs = [0 for _ in range(self.output_number)]
                expected_outputs[data[-1]] = 1
                error_sum += sum([(expected_outputs[i] - outputs[i]) ** 2 for i in range(len(expected_outputs))])
                self.backward_propagation(expected_outputs)
                self.update_weights(data, learn_rate)
            print(f"epoch: {epoch}, E: {error_sum}")

    def predict(self, data):
        outputs = self.forward_propagation(data)
        return outputs.index(max(outputs))

"""
train_set = Data(filename = 'winequality-red.csv', has_header = True, delimiter_type= ";")
train_set.prepare_data()
print(train_set.dataset)
"""



"""
data = [
    [x1,x2,x3,y], - pierwsza próbka
    [x1,x2,x3,y] - druga próbka itd.
]

network = NeuralNetwork(3,3,3)
network.train(1,1,data)
network.predict(datax)
"""
