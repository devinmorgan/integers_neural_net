import numpy as np
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = []
        hidden_layers = sizes[1:]
        for layer_size in hidden_layers:
            self.biases.append(
                np.random.randn(layer_size, 1)
            )

        self.weights = []
        for i in range(self.num_layers - 1):
            self.weights.append(
                np.random.randn(sizes[i], sizes[i+1])
            )

    def feed_forward(self, a):
        for i in range(self.num_layers):
            b = self.biases[i]
            w = self.weights[i]
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        num_training_data = len(training_data)
        num_tests = 0 if test_data is None else len(test_data)

        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = []
            for k in xrange(0, num_training_data, mini_batch_size):
                mini_batches.append(
                    training_data[k:k + mini_batch_size]
                )

            for mini_batch in mini_batches:
                self.update_network_with_mini_batch(mini_batch, learning_rate)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), num_tests)
            else:
                print "Epoch {0} complete".format(j)

    def update_network_with_mini_batch(self, mini_batch, learning_rate):
        gradient_b = []
        for b in self.biases:
            gradient_b.append(
                np.zeros(b.shape)
            )

        gradient_w = []
        for w in self.weights:
            gradient_w.append(
                np.zeros(w.shape)
            )

        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)

            updated_gradient_b = []
            for gr_b, delta_gr_b in zip(gradient_b, delta_gradient_b):
                updated_gradient_b.append(
                    gr_b + delta_gr_b
                )
            gradient_b = updated_gradient_b

            updated_gradient_w = []
            for gr_w, delta_gr_w in zip(gradient_w, delta_gradient_w):
                updated_gradient_w.append(
                    gr_w + delta_gr_w
                )
            gradient_w = updated_gradient_w

            updated_weights = []
            for w, gr_w in zip(self.weights, gradient_w):
                updated_weights.append(
                    w - (learning_rate / len(mini_batch)) * gr_w
                )
            self.weights = updated_weights

            updated_biases = []
            for b, gr_b in zip(self.biases, gradient_b):
                updated_weights.append(
                    b - (learning_rate / len(mini_batch)) * gr_b
                )
            self.biases = updated_biases







































