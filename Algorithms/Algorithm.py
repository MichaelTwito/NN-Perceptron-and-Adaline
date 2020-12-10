# from abc import ABC, abstractmethod

class Algorithm(object):
    """
    This class represents an Algorithm.
    """

    def __init__(self):
        self.x = 4
        self.y = 5
       


    # @abstractmethod
    def train(self,data,label):
        pass

    # @abstractmethod
    def test(self,data,label):
        pass