import data_loader
import numpy as np, random, sys
from past.builtins import xrange
class Network(object):

    def __init__(self, sizes):
        # sizes is a list containing the size of each layer, like [50, 20, 10]
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        # creating biases for all layers except the input layer. list of matrices
        self.biases=[]
        for layer in sizes[1:]:
            self.biases.append(np.random.randn(layer, 1))
        #iterating between layers and assigning weights(omitting the first and last layer)
        self.weights = []
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))
    
    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        for item in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = []
            for index in xrange(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[index : index + mini_batch_size])
        

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch , eta)
            if test_data:
                print("epoch {0}: {1} / {2}".format(item, self.evaluate(test_data), len(test_data)))
            else:
                print("epoch {0} complete".format(item))
        return (self.weights, self.biases)

    def update_mini_batch(self, mini_batch, eta):
        # updates weights and biases after gradient descent/backprop
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
   
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b , delta_nabla_b)] #nabla b + delta nabla b
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w , delta_nabla_w)] #nabla w + delta nabla w
            
        new_biases=[]
        for b, nb in zip(self.biases, nabla_b):
            new_biases.append(b-(eta/len(mini_batch))*nb) #formula
        self.biases = new_biases

        new_weights = []
        for w, nw in zip(self.weights, nabla_w):
            new_weights.append(w-(eta/len(mini_batch))*nw)
        self.weights = new_weights

    def backprop(self, x, y):
        #returns tuple (nabla_b, nabla_w) where they are both numpy arrays. they depend on Cost(x)
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        #feeding forward
        activation = x
        activation_list = [x] # we want a list of activations
        z_vector_list = [] # we want a list of z vectors, layer by layer
        for bias, weight in zip(self.biases, self.weights):
            
            z = np.dot(weight, activation)+bias
            z_vector_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
            # in short, a(sub n+1) = sigmoid(w . a(sub n) + b)

        #backpropogation
        delta = self.cost_derivative(activation_list[-1], y) * sigmoid_derivative(z_vector_list[-1])
        #delta = change in last layer * sigmoid(z) in last layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation_list[-2].transpose())
        # iterating throught the network backward(referencing negative indeces)
        for layer in xrange(2, self.number_of_layers):
            z = z_vector_list[-layer]
            sigmoid_der = sigmoid_derivative(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sigmoid_der #backpropogating delta
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activation_list[-layer-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = []
        for x, y in test_data:
            test_results.append((np.argmax(self.feedforward(x)), y))
            #highest activation in final layer
        sum = 0
        for x,y in test_results:
            if int(x)==int(y):
                sum+=1
        return sum


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

# returns sigmoid or derivative of sigmoid
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1-x)
    
#use to run the code 
#if __name__ == "__main__":
#   training_data , validation_data , test_data = data_loader.load_the_data_wrap()
#   net = Network([784, 30, 10])
#   net.SGD(training_data , 30, 10, 3.0, test_data=test_data)
    
