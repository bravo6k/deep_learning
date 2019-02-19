import numpy as np
import math
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

def dictionary_to_vector(parameters):

    keys = []
    count = 0
    for key in parameters.keys():
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys.extend([key]*new_vector.shape[0])
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(old_p,_,vec):

    parameters = {}
    for i in np.unique(_):
        index = np.where(np.isin(_, i))[0]
        parameters[i] = vec[index].reshape(old_p[i].shape)

    return parameters

def gradients_to_vector(gradients):

    count = 0
    for key in gradients.keys():
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradient_check(parameters, gradients, X, Y, act_fun_list, keep_prob, lambd, epsilon = 1e-7):

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        thetaplus = np.copy(parameters_values)                                      
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                
        new_parameters = vector_to_dictionary(parameters,_,thetaplus)
        AL, c = forward_propagation(act_fun_list=act_fun_list, parameters = new_parameters, X=X, keep_prob = keep_prob)
        J_plus[i] = compute_cost(AL=AL,Y=Y, lambd = lambd, parameters = new_parameters)      

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                    
        thetaminus[i][0] = thetaminus[i][0] - epsilon   
        new_parameters = vector_to_dictionary(parameters,_,thetaminus)
        AL, c = forward_propagation(act_fun_list=act_fun_list, parameters=new_parameters, X=X, keep_prob = keep_prob)  
        J_minus[i] = compute_cost(AL=AL,Y=Y, lambd = lambd, parameters = new_parameters) 

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)                                     
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   
    difference = numerator / denominator                                              


    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


def random_mini_batches(X, Y, mini_batch_size):

    n = X.shape[1]
    n_class = Y.shape[0]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(n))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((n_class,n))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(n/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if n % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size*num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*num_complete_minibatches:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


class Neural_Network:

    def __init__(self, layer_list, activation_function, lambd = 0, keep_prob = 1, mini_batch_size = 256, learning_rate=0.001):
        self.layer_list = layer_list
        self.act_fun_list = activation_function
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.batch_size = mini_batch_size

    def train(self, X, Y, epoch = 300, print_cost=False):

        costs = []
        # Parameters initialization
        parameters = initialize_parameters(layer_list=self.layer_list, act_fun_list = self.act_fun_list)

        # Loop (gradient descent)
        for i in range(0, epoch):
            
            mini_batches = random_mini_batches(X=X, Y=Y, mini_batch_size = self.batch_size)

            for minibatch in mini_batches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation:
                AL, caches = forward_propagation(act_fun_list=self.act_fun_list, parameters=parameters, X=minibatch_X, keep_prob = self.keep_prob)

                # Compute cost.
                cost = compute_cost(AL=AL,Y=minibatch_Y, lambd = self.lambd, parameters = parameters)

                # Backward propagation.
                grads = backward_propagation(act_fun_list=self.act_fun_list, cache=caches, Y=minibatch_Y, parameters=parameters, lambd=self.lambd, layer_list=self.layer_list, keep_prob=self.keep_prob)

                # Update parameters.
                parameters = update_parameters(parameters=parameters, grads=grads, learning_rate = self.learning_rate)
                
                diff = gradient_check(parameters, grads, minibatch_X, minibatch_Y, self.act_fun_list, self.keep_prob, self.lambd, epsilon = 1e-7)
                
            # Print the cost every 50 training example
            if print_cost:
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
