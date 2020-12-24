import numpy as np


class BackPropogation(object):
    """
        This class represents an Perceptron algorithm.
    """
    def __init__(self,hiddenSize):
        #parameters
        self.inputSize = 33
        self.outputSize = 2
        self.hiddenSize = hiddenSize
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #  weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #  weight matrix from hidden to output layer
        
    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights 
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights 
        output = self.sigmoid(self.z3)
        return output
        
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return 0.5*(1 + self.sigmoid(s))*(1 - self.sigmoid(s))
        return (2/(1 + np.exp(-s))) - 1
    
    def backward(self, X, y, output):
        #backward propogate through the network

        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error   
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        