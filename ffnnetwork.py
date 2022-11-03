import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        Ein Netzwerk besteht aus folgende Sachen:
        num_layers: Anzahl der Layers
        sizes: Anzahl der Neuronen, die sich in verschiedene Layer befinden.
        Beispiel: [2, 3, 1] (2 Neuronen im Input Layer, 3 im Hidden Layer und
        1 im Output Layer)
        biases: Die zugehörigen Biases im Hidden und Output Layer
        weights: Die zugehörigen Gewichte im Hidden und Output Layer
        Bei beiden Parameter werden erstmal zufällige Werte gesetzt.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Gibt das Ergebnis des Netzwerkes wieder, wenn "a" die Eingabe wäre
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Hier wird der Stochastic Gradient Ascent benutzt.
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {} Accuracy: {}".format(j,self.evaluate(test_data),n_test, self.evaluate(test_data)/n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Aktualisiert die Gewichte w und Biases b des Netzwerkes mithilfe
        von Stochastic Gradient Descent.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Dies ist eine Methode, die sich Backpropagation nennt.
        Sie ist dafür da, um die partielle Ableitung
        des Stochastic Gradient Ascent zu berechnen.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feed Forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Back Propagation
        delta = self.loss_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Hier werden die Ergebnisse zurückgegeben. Dabei zählen
        nur die richtigen Ergebnisse gezählt, die getestet wurden
        (nicht die, die trainiert wurden).
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def loss_derivative(self, output_activations, y):
        """
        Gibt den Wert der abgeleitete Loss Function zurück.
        """
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """
    Sigmoid Funktion
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    Ableitung der Sigmoid Funktion
    """
    return sigmoid(z)*(1-sigmoid(z))