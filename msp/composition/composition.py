from ase.data import chemical_symbols
import numpy as np
import random

def generate_random_compositions(dataset, n=5, max_elements=5, max_atoms=10):
    """
    Generate n unique compositions that do not appear in dataset randomly

    Args:
        dataset (dict): dictionary of dataset
        n (int): number of compositions to generate
        max_elements (int): maximum number of elements in composition
        max_atoms (int): maximum number of atoms per element
    
    Returns:
        compositions (list): list of compositions
    """
    compositions = []
    dataset_comps = []
    for data in dataset:
        data['atomic_numbers'].sort()
        dataset_comps.append(data['atomic_numbers'])
    for i in range(n):
        while True:
            comp = []
            num_atoms = np.random.randint(1, max_elements + 1)
            random_atoms = np.random.randint(1, 101, size=num_atoms)
            random_atoms = np.sort(random_atoms)
            for atom in random_atoms:
                comp.extend([atom] * np.random.randint(1, max_atoms))
            if comp not in dataset_comps:
                compositions.append(comp)
                break
    return compositions

def sample_random_composition(dataset, n=5):
    dataset_comps = []
    for data in dataset:
        data['atomic_numbers'].sort()
        dataset_comps.append(data['atomic_numbers'])
    return [dataset_comps[i] for i in random.sample(range(len(dataset_comps)), n)]

