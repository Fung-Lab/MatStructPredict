from ase import Atoms
import numpy as np

class Optimizer:
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
    
    def optimize(self, composition, cell, topk):
        """
        Optimizes the composition using the optimizer
        
        Args:
            composition (str): A string representing a chemical composition
        
        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        pass

    def atom_from_str(self, composition, cell):
        """
        Creates an ASE atoms object from a composition string
        
        Args:
            composition (str): A string representing a chemical composition
        
        Returns:
            ase.Atoms: An ASE atoms object representing the composition
        """
        elements = []
        counts = []
        current_element = ''
        current_count = ''
        for char in composition:
            if char.isalpha():
                if current_count != '':
                    counts.append(int(current_count))
                    current_count = ''
                current_element += char
            elif char.isdigit():
                if current_element != '':
                    elements.append(current_element)
                    current_element = ''
                current_count += char

        if current_count != '':
            counts.append(int(current_count))

        for i, element in enumerate(elements):
            elements.extend([element] * (counts[i] - 1))

        atoms = Atoms(elements, cell=cell, pbc=(True, True, True))
        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)
        return atoms
