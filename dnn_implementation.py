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
        A = act_fun_list[i-1].get_result(Z)
        cache['Z'+str(i)] = Z
        cache['A'+str(i)] = A
        
