import numpy as np
from activation_function import *





def initialize_parameters(layer_list, custom_initial_parameter):

    n_layer = len(layer_list)
    parameters = {}

    for i in range(1,n_layer):
        parameters['W'+str(i)] = np.random.randn(layer_list[i],layer_list[i-1])*custom_initial_parameter
        parameters['b'+str(i)] = np.zeros([layer_list[i],1])

        assert(parameters['W' + str(i)].shape == (layer_list[i], layer_list[i-1]))
        assert(parameters['b' + str(i)].shape == (layer_list[i], 1))

    return parameters


def forward_propagation(act_fun_list, parameters, X):

    n_layer = len(parameters) // 2
    if n_layer != len(act_fun_list)-1:
        print("Error: Number of activation functions don't match with number of layers.")

    A = X
    cache = {}
    for i in range(1, n_layer):
        A_prev = A
        Z = np.dot(parameters['W'+str(i)], A) + parameters['b'+str(i)]
        A = act_fun_list[i-1]().get_result(Z)
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A

    A = Softmax().get_result(A)
    cache['softmax'] = A
    return A, cache


def compute_cost(AL, Y):

    n = Y.shape[1]
    cost = -1/n*np.sum(np.log(AL)*Y)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def backward_propagation(cache, Y):

    grads = {}
    L = len(caches)//2 # the number of layers
    n = Y.shape[1]
    A_softmax = cache['softmax']
    Y = Y.reshape(A_softmax.shape) # after this line, Y is the same shape as AL

    dAL = A_softmax - Y
    
