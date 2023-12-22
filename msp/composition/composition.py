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
    composition_strings = []
    for i in range(n):
        while True:
            composition_string = ''
            composition_dict = {}
            random_symbols = np.random.choice(chemical_symbols[1:100], size=np.random.randint(1, max_elements + 1), replace=False)
            random_symbols = np.sort(random_symbols)
            for sym in random_symbols:
                count = np.random.randint(1, max_atoms + 1)
                composition_string += sym + str(count) + ' '
                composition_dict[sym] = count
            composition_string = composition_string[:-1]
            if composition_string not in composition_strings and composition_string not in dataset:
                composition_strings.append(composition_string)
                compositions.append(composition_dict)
                break
    return compositions