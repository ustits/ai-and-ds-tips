import math
import numpy as np


# функция активая - пороговая функция
def step_function(x):
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def perceptron_output(weights, bias, activation, inputs):
    calculation = np.dot(weights, inputs) + bias
    return activation(calculation)


# логическая схема И
weights = np.array([2, 2])
bias = -3

print(perceptron_output(weights, bias, step_function, np.array([1, 1])))
print(perceptron_output(weights, bias, step_function, np.array([0, 0])))
print(perceptron_output(weights, bias, step_function, np.array([0, 1])))

# логическое ИЛИ
bias = -1
print(perceptron_output(weights, bias, step_function, np.array([1, 1])))
print(perceptron_output(weights, bias, step_function, np.array([0, 0])))
print(perceptron_output(weights, bias, step_function, np.array([0, 1])))

# логическое НЕ
weights = np.array([-2])
bias = 1
print(perceptron_output(weights, bias, step_function, np.array([1])))
print(perceptron_output(weights, bias, step_function, np.array([0])))


def feed_forward(neural_network, input_vector):
    outputs = []
    output = None
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [perceptron_output(neuron, 0, sigmoid, input_with_bias) for neuron in layer]
        outputs.append(output)

    input_vector = output
    return outputs

xor_network = np.array([
    [
        [20, 20, -30],
        [20, 20, -10]
    ],
    [
        [-60, 60, -30]
    ]
])

print(0, 0, feed_forward(xor_network, [0, 0]))