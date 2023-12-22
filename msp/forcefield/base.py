from abc import ABC, abstractmethod

class ForceField(ABC):

    @abstractmethod
    def __init__(self):
        """
        Initialize the surrogate model.
        """
        pass
    
    @abstractmethod
    def train(self, dataset):
        """
        Train the surrogate model on the dataset.
        """
        pass

    @abstractmethod
    def update(self, dataset):
        """
        Update the surrogate model on the dataset.
        """
        pass