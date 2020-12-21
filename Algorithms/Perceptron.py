# from Algorithms.Algorithm import Algorithm
import numpy as np


class Perceptron(object):
    """
        This class represents an Perceptron algorithm.
    """

    def __init__(self, dim=2, eta=1, max_epoch=None):
        self.dim = dim
        self.W = np.zeros(dim)
        self.b = np.zeros(1)
        self.eta = eta
        self.max_epoch = max_epoch

    def train(self, data, label, logs=False):
        i = 0
        count = 0
        finished = True
        while count != data.shape[0] and self.max_epoch is None:
            count += 1
            if label[i] * (np.sum(self.W * data[i, :], axis=-1) + self.b)[0] <= 0:
                self.W += self.eta * label[i] * data[i, :]
                self.b += self.eta * label[i]
            i = (i + 1) % data.shape[0]
        print("train")

    def predict(self, data):
        return np.sign((np.sum(self.W*data, axis=-1)+self.b))

    def score(self, data, labels):
        if x.shape[-1] != self.dim:
            print("The input shape is incorrect!")
            return 0
        dis = np.abs(np.sum(self.W * data, axis=-1) + self.b) * labels
        return -np.sum(dis * (dis < 0)) * 1 / np.norm(self.W)