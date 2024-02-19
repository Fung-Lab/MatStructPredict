from abc import ABC, abstractmethod

class Optimizer(ABC):
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
    
    @abstractmethod
    def predict(self, composition, cell, topk):
        """
        Optimizes the composition using the optimizer
        
        Args:
            composition (str): A string representing a chemical composition
        
        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        pass

