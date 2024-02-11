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

    def atom_from_comp(self, composition, density=.2):
        """
        Creates an ASE atoms object from a composition list
        
        Args:
            composition (list): A list representing the atomic numbers
        
        Returns:
            ase.Atoms: An ASE atoms object representing the composition
        """
        beta = np.random.uniform(0, 180)
        gamma = np.random.uniform(0, 180)
        minCosA = - np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        maxCosA = np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        alpha = np.random.uniform(minCosA, maxCosA)
        alpha = np.arccos(alpha) * 180 / np.pi
        a = np.random.rand() + .000001
        b = np.random.rand() + .000001
        c = np.random.rand() + .000001
        cell=[a, b, c, alpha, beta, gamma]
        atoms = Atoms(composition, cell=cell, pbc=(True, True, True))
        vol = atoms.get_cell().volume
        ideal_vol = len(composition) / density
        scale = (ideal_vol / vol) ** (1/3)
        cell = [scale * a, scale * b, scale * c, alpha, beta, gamma]
        atoms.set_cell(cell)
        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)
        return atoms

    def atoms_to_dict(self, atoms, loss):
        """
        Creates a list of dict from a list of ASE atoms objects
        
        Args:
            atoms (list): A list of ASE atoms objects
            energy (list): A list of predicted energies for each ASE atoms object.
        
        Returns:
            list: Contains atoms represented as dicts
        """
        res = [{} for _ in atoms]
        for i, d in enumerate(res):
            d['n_atoms'] = len(atoms[i].get_atomic_numbers())
            d['pos'] = atoms[i].get_positions()
            d['cell'] = atoms[i].get_cell()
            d['z'] = atoms[i].get_atomic_numbers()
            d['atomic_numbers'] = atoms[i].get_atomic_numbers()
            d['loss'] = loss[i]
        return res
