# from Algorithms.Algorithm import Algorithm
import numpy as np


class Perceptron(object):
    """
        This class represents an Perceptron algorithm.
    """

    def __init__(self, dim=2, eta=1, max_epoch=None):
        self.dim = dim
        self.W = np.zers(dim)
        self.b = np.zeros(1)
        self.eta = eta
        self.max_epoch = 1000

    def train(self, data, label, logs=False):
        i = 0
        count = 0
        epoch = 0
        finished = True
        while count != data.shape[0] and (self.max_epoch is None or epoch > self.max_epoch):
            count += 1
            if label[i] * (np.sum(self.W * data[i, :], axis=-1) + self.b) <= 0:
                self.W += self.eta * label[i] * data[i, :]
                self.b += self.eta * label[i]
                epoch += 1
                if logs:
                    print("Epoch:", epoch, ": Weights", self.W, " Bias:", self.b)
            i = (i + 1) % data.shape[0]
        print("train")

    def test(self, data, label):
        print("train")
