import random
import numpy as np

# Implementation of simple neural network from scratch
# This will be used for both hw3 and milestone 2 of the 
# final project 

# Let's define the Network class for the neural network 
class Network:
    """"
    The constructor takes one parameter, sizes. The sizes variable is a list of numbers that indicates the number of 
    input nodes at each layer in our neural network. In our __init__ function, we initialize four attributes. 
    The number of layers, num_layers, is set to the length of the sizes and the list of the sizes of the 
    layers is set to the input variables, sizes. Next, the initial biases of our network are randomized 
    for each layer after the input layer. Finally, the weights connecting each node are randomized for each 
    connection between the input and output layers. For context, np.random.randn() returns a random sample from 
    the normal distribution.
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]                         # initialize the biases randomly
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]    # initialize the weights randomly
    
    # Helper functions: let's define the activation function and its derivative for each node
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    """
    The feed_forward function is the function that sends information forward in the neural network. 
    This function will take one parameter, x, representing the current activation vector. 
    This function loops through all the biases and weights in the network and calculates the activations at 
    each layer. The x returned is the activations of the last layer, which is the prediction,i.e., y = w * x + b
    """    
    def feed_forward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    """
    stochastic_gradient_decent function is the function that performs the training job. 
    we are doing an altered version of gradient descent known as mini-batch (stochastic) gradient descent. 
    This means that we are going to update our model using a mini-batch of data points. This function takes 
    four mandatory parameters and one optional parameter. The four mandatory parameters are the set of training 
    data, the number of epochs, the size of the mini-batches, and the learning rate (eta). We can optionally 
    provide test data. When we test this network later, we will provide test data.
    """
    def stochastic_gradient_decent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        samples = len(training_data)
       
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
       
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # print the log if there is no test data provided
            if not test_data:
                print(f"Epoch {j} complete")
            # print the accuracy using test data (if any) after training each epoch
            else:
                if j == epochs - 1:
                    print(f"Epoch {j}: accuracy = {self.evaluate(test_data)} / {n_test}")
    
    # Helper function which is used in backpropagation
    def cost_derivative(self, output_activations, y):
        return(output_activations - y)
    
    """
    We start our backward pass by calculating the delta, which is equal to the error from the last layer multiplied 
    by the sigmoid_prime of the last entry of the zs vectors. We set the last layer of nabla_b as the delta and the 
    last layer of nabla_w equal to the dot product of the delta and the second to last layer of activations 
    (transposed so we can actually do the math). After setting these last layers up, we do the same thing for each 
    layer going backwards starting from the second to last layer. Finally, we return the nablas as a tuple.
    """
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (nabla_b, nabla_w)
    
    """
    Mini-batch updating is part of our SGD (stochastic) gradient descent function from earlier. It starts much the 
    same way as our backprop function by creating 0 vectors of the nablas for the biases and weights. It takes two parameters, 
    the mini_batch, and the learning rate, eta. Then, for each input, x, and output, y, in the mini_batch, we get the 
    delta of each nabla array via the backprop function. Next, we update the nabla lists with these deltas. 
    Finally, we update the weights and biases of the network using the nablas and the learning rate. Each 
    value is updated to the current value minus the learning rate divided by the size of the minibatch times the nabla value.
    """
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    """
    This function takes one parameter, the test_data. In this function, we simply compare the network outputs 
    with the expected output, y. The network outputs are calculated by feeding forward the input, x. This simply represents accuracy!
    """     
    def evaluate(self, test_data):
        accuracy = 0
        for (x, y) in test_data:
            if np.argmax(self.feed_forward(x)) == np.argmax(y):
                accuracy += 1
        return accuracy

# helper function to encode the labels as 0 or 1
def one_hot_encode(y):
    encoded = np.zeros((2, 1))
    encoded[y] = 1.0
    return encoded        

x = np.array([
  [0, 0],
  [0, 1], 
  [1, 0],
  [1, 1]
])

y_AND = np.array([0, 0, 0, 1])
y_OR = np.array([0, 1, 1, 1])
y_XOR = np.array([0, 1, 1, 0])

a = np.array([
  [0, 0],
  [0, 1], 
  [1, 0],
  [1, 1]
])

b_AND = np.array([0, 0, 0, 1])
b_OR = np.array([0, 1, 1, 1])
b_XOR = np.array([0, 1, 1, 0])

train_x = [np.reshape(i, (2, 1)) for i in x]
train_y_AND = [one_hot_encode(j) for j in y_AND]
train_y_OR = [one_hot_encode(j) for j in y_OR]
train_y_XOR = [one_hot_encode(j) for j in y_XOR]
test_x = [np.reshape(i, (2, 1)) for i in a]
test_y_AND = [one_hot_encode(j) for j in b_AND]
test_y_OR = [one_hot_encode(j) for j in b_OR]
test_y_XOR = [one_hot_encode(j) for j in b_XOR]

training_data_AND = zip(train_x, train_y_AND)
testing_data_AND = zip(test_x, test_y_AND)

training_data_OR = zip(train_x, train_y_OR)
testing_data_OR = zip(test_x, test_y_OR)

training_data_XOR = zip(train_x, train_y_XOR)
testing_data_XOR = zip(test_x, test_y_XOR)

if __name__ == "__main__":
    print("======================================")
    print("Trainig and inference accuracy for different logics\n")
    print("AND logic\n")
    net_and = Network([2, 6, 2])
    net_and.stochastic_gradient_decent(training_data_AND, 100, 4, 5.0, testing_data_AND)

    print("======================================")
    print("OR logic\n")
    net_or = Network([2, 6, 2])
    net_or.stochastic_gradient_decent(training_data_OR, 100, 4, 5.0, testing_data_OR)


    print("======================================")
    print("XOR logic\n")
    net_xor = Network([2, 6, 2])
    net_xor.stochastic_gradient_decent(training_data_XOR, 100, 4, 5.0, testing_data_XOR)

    print("======================================")
    print("For the analyzed logics, having one hidden layer with 6 nodes and 100 epochs gives 100% accuracy for all logics")