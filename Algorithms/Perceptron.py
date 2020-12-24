# from Algorithms.Algorithm import Algorithm
import numpy as np


class Perceptron(object):
    """
        This class represents an Perceptron algorithm.
    """

    def __init__(self, dim=2, eta=0.01, max_epoch=10000):
        self.dim = dim
        self.W = np.random.rand(dim)
        self.b = np.zeros(1)
        self.eta = eta
        self.max_epoch = max_epoch

    def train(self, data, label, logs=False):
        for epoch in range(0,self.max_epoch):
            for data_item, label_item in zip(data, label):
                if label_item * (np.sum(self.W * data_item) + self.b)[0]<0:
                    self.W += self.eta * label_item * data_item
                    self.b += self.eta * label_item


    def predict(self, data):
        return np.sign((np.sum(self.W*data, axis=-1)+self.b))

    def score(self,test_data,test_labels):
        count = 0
        for test,label in zip(test_data,test_labels):
            if self.predict(test) != label:
                count +=1
        print('total misses: '+str(count))
