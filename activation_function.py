import numpy as np

class Sigmoid:

    def __init__(self):
        pass

    def get_result(self, Z):
        return 1/(1+np.exp(-Z))

    def prime(self, Z):
        return self.get_result(Z)*(1 - self.get_result(Z))
    
class Tanh:

    def __init__(self):
        pass

    def get_result(self,Z):
        return (np.exp(2*Z)-1)/(np.exp(2*Z)+1)

    def prime(self,Z):
        return 1 - self.get_result(Z)**2

class ReLU:

    def __init__(self):
        pass

    def get_result(self, Z):
        return np.maximum(0,Z)

    def prime(self, Z):
        return np.where(Z>0,1,0)

class Leaky_ReLU:

    def __init__(self):
        pass

    def get_result(self, Z):
        return np.maximum(-0.01*Z,Z)

    def prime(self, Z):
        return np.where(Z>0,1,-0.01)

class Softmax:

    def __init__(self):
        pass

    def get_result(self,Z):
        exp = np.exp(Z)
        return exp/np.sum(exp,axis=0,keepdims=True)

