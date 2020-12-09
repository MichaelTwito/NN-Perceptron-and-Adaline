from abc import ABC, abstractmethod

class Algorithm(ABC):
    """
    This class represents an Algorithm.
    """

    @abstractmethod
    def train(self,data,label):
        pass

    @abstractmethod
    def test(self,data,label):
        pass