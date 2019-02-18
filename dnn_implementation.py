import numpy as np
from activation_function import *





def initialize_parameters(layer_list, act_fun_list):

    n_layer = len(layer_list)
    parameters = {}

    weight_init = [np.sqrt(2/layer_list[j]) if i == ReLU else np.sqrt(1/layer_list[j]) if i == Tanh else np.sqrt(2/layer_list[j]+layer_list[j+1]) for j,i in enumerate(act_fun_list)]
    for i in range(1,n_layer):
        parameters['W'+str(i)] = np.random.randn(layer_list[i],layer_list[i-1])*0.01
        parameters['b'+str(i)] = np.zeros([layer_list[i],1])

        assert(parameters['W' + str(i)].shape == (layer_list[i], layer_list[i-1]))
        assert(parameters['b' + str(i)].shape == (layer_list[i], 1))

    return parameters


def forward_propagation(act_fun_list, parameters, X, keep_prob):

    n_layer = len(parameters) // 2
    if n_layer != len(act_fun_list):
        print("Error: Number of activation functions don't match with number of layers.")

    A = X
    cache = {}
    cache['A0'] = A
    for i in range(1, n_layer):
        A_prev = A
        Z = np.dot(parameters['W'+str(i)], A_prev) + parameters['b'+str(i)]
        A = act_fun_list[i-1]().get_result(Z)
        D = np.random.rand(A.shape[0],A.shape[1])
        D = D<keep_prob
        A = A*D
        A = A/keep_prob
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A
        cache['D'+str(i)] = D

    # last layer
    Z = np.dot(parameters['W'+str(n_layer)], A) + parameters['b'+str(n_layer)]
    A = act_fun_list[n_layer-1]().get_result(Z)
    cache['Z'+str(n_layer)] = Z
    cache['A'+str(n_layer)] = A

    A = Softmax().get_result(A)
    cache['softmax'] = A

    return A, cache


def compute_cost(AL, Y, lambd, parameters):

    n = Y.shape[1]
    W = [parameters[i] for i in parameters.keys() if 'W' in i]
    L2_regularization_cost = 1/n*lambd/2*(np.sum(np.concatenate(np.square(W),axis=None)))
    cost = -1/n*np.sum(np.log(AL)*Y)+L2_regularization_cost
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def backward_propagation(cache, Y, act_fun_list, parameters,lambd, layer_list,keep_prob):

    grads = {}
    L = len(layer_list) - 1 # the number of hidden layers
    n = Y.shape[1]
    A_softmax = cache['softmax']
    Y = Y.reshape(A_softmax.shape) # after this line, Y is the same shape as AL

    dAL = A_softmax - Y

    for i in reversed(range(1,L+1)):

        grads['dA'+str(i)] = dAL

        dZ = dAL * act_fun_list[i-1]().prime(cache['Z'+str(i)]) # dZL = dAL*g'(ZL)
        dW = 1/n*np.dot(dZ,cache['A'+str(i-1)].T)+lambd/n*parameters['W'+str(i)] # dW = 1/n* dZ %*% AL-1.T * lamdba/n * W
        db = 1/n*np.sum(dZ,axis = 1, keepdims = True)
        if i!= 1:
            dAL = np.dot(parameters['W'+str(i)].T,dZ)
            dAL = dAL*cache['D'+str(i-1)]
            dAL = dAL/keep_prob

        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


class Neural_Network:

    def __init__(self, layer_list, activation_function, lambd = 0, keep_prob = 1, learning_rate=0.01):
        self.layer_list = layer_list
        self.act_fun_list = activation_function
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.keep_prob = keep_prob

    def train(self, X, Y, num_iterations = 300, print_cost=False):

        costs = []

        # Parameters initialization
        parameters = initialize_parameters(layer_list=self.layer_list, act_fun_list = self.act_fun_list)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation:
            AL, caches = forward_propagation(act_fun_list=self.act_fun_list, parameters=parameters, X=X, keep_prob = self.keep_prob)

            # Compute cost.
            cost = compute_cost(AL=AL,Y=Y, lambd = self.lambd, parameters = parameters)

            # Backward propagation.
            grads = backward_propagation(act_fun_list=self.act_fun_list, cache=caches, Y=Y, parameters=parameters, lambd=self.lambd, layer_list=self.layer_list, keep_prob=self.keep_prob)

            # Update parameters.
            parameters = update_parameters(parameters=parameters, grads=grads, learning_rate = self.learning_rate)

            # Print the cost every 50 training example
            if print_cost and i % 50 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if i % 50 == 0:
                costs.append(cost)

        self.costs = costs
        self.parameters = parameters

    def predict_prob(self, X):
        A, cache = forward_propagation(self.act_fun_list, self.parameters, X)
        return A

    def predict(self, X):
        A, cache = forward_propagation(self.act_fun_list, self.parameters, X)
        result = np.argmax(A,axis=0)
        return result
