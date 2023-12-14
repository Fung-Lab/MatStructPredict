from ase.data import chemical_symbols
import numpy as np

def sample_composition_random(dataset, n=5, max_elements=5, max_atoms=10):
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
    for i in range(n):
        while True:
            composition = ''
            random_symbols = np.random.choice(chemical_symbols, size=np.random.randint(1, max_elements + 1), replace=False)
            random_symbols = np.sort(random_symbols)
            for sym in random_symbols:
                composition += sym + str(np.random.randint(1, max_atoms + 1)) + ' '
            composition = composition[:-1]
            if composition not in compositions and composition not in dataset:
                compositions.append(composition)
                break
    return compositions