from ase import Atoms
from ase.data import atomic_numbers
from abc import ABC, abstractmethod
import numpy as np

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

    def atom_from_dict(self, composition, cell):
        """
        Creates an ASE atoms object from a composition string
        
        Args:
            composition (dict): A dictionary representing a chemical composition
        
        Returns:
            ase.Atoms: An ASE atoms object representing the composition
        """
        elements = []
        for el in composition:
            elements.extend([atomic_numbers[el]] * composition[el])
        atoms = Atoms(elements, cell=cell, pbc=(True, True, True))
        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)
        return atoms
