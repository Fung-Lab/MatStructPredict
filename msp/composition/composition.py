from ase.data import chemical_symbols
import numpy as np
import random
import math
from msp.structure.structure_util import smact_validity
from collections import Counter

def hash_structure(atomic_numbers):
    """
    Hashes the atomic numbers of a structure
    Args:
        atomic_numbers (list): list of atomic numbers
    
    Returns:
        hash (int): hash of atomic numbers
    """
    counts = Counter(atomic_numbers)
    sorted_counts = sorted(counts.items())
    # divide the counts by the gcd of the counts
    gcd = sorted_counts[0][1]
    for elem in sorted_counts:
        gcd = math.gcd(gcd, elem[1])
    tuple_sorted_counts = tuple([(elem[0], elem[1]//gcd) for elem in sorted_counts])
    return hash(tuple_sorted_counts)

def hash_dataset(dataset):
    """
    Hashes the structures in the dataset
    Args:
        dataset (dict): dictionary of dataset
    
    Returns:
        hashed_structures (dict): dictionary of hashed structures
    """
    hashed_structures = {}
    for data in dataset:
        hashed_structures[hash_structure(data['atomic_numbers'])] = data
    return hashed_structures

def generate_random_compositions(dataset, n=5, max_elements=5, max_atoms=20, elems_to_sample=[]):
    """
    Generate n unique compositions that do not appear in dataset randomly
    Args:
        dataset (dict): dictionary of dataset
        n (int): number of compositions to generate
        max_elements (int): maximum number of unique elements in composition
        max_atoms (int): maximum number of atoms in composition
        elems_to_sample (list): list of elements to sample from
    
    Returns:
        compositions (list): list of compositions
    """
    compositions = []
    comp_hashes = []
    hashed_dataset = hash_dataset(dataset)
    if len(elems_to_sample) == 0:
        elems_to_sample = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 
                          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
                          48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    for i in range(n):
        while True:
            comp = []
            num_unique = int(np.round(np.random.normal(3, 1)))
            while num_unique < 1 or num_unique > max_elements:
                num_unique = int(np.round(np.random.normal(3, 1)))
            rand_elems = np.random.choice(elems_to_sample, num_unique, replace=False)
            temp = max_atoms + 1
            freq = []
            while True:
                freq = []
                for elem in rand_elems:
                    if temp == 2:
                        freq.append(2)
                        break
                    elif temp <= 1:
                        freq.append(max_atoms)
                        break
                    f = np.random.randint(2, temp)
                    freq.append(f)
                    temp -= f
                    comp.extend([elem] * f)
                if sum(freq) <= max_atoms:
                    break
                print('Invalid composition: ', comp)
            print('Potential composition: ', comp)
            smact_valid = smact_validity(rand_elems, freq)
            print('SMACT validity: ', smact_valid)
            if not smact_valid:
                print('Invalid composition')
                continue
            comp_hash = hash_structure(comp)
            if comp_hash not in hashed_dataset and comp_hash not in comp_hashes:
                print('Accepted composition', i, ':', comp)   
                compositions.append(comp)
                comp_hashes.append(comp_hash)
                break
            else:
                print('Invalid compositon, already occurs in dataset')
    return compositions

def generate_random_lithium_compositions(dataset, n=5,  max_elements=6, max_atoms=20, li_ratio_lower=.2, li_ratio_upper=.4, halide_ratio_lower=.2, halide_ratio_upper=.5):
    """
    Generate n unique lithium compositions that do not appear in dataset randomly
    Args:
        dataset (dict): dictionary of dataset
        n (int): number of compositions to generate
        max_elements (int): maximum number of unique elements in composition
        max_atoms (int): maximum number of atoms in composition
        li_ratio_lower (float): lower bound for lithium ratio
        li_ratio_upper (float): upper bound for lithium ratio
        halide_ratio_lower (float): lower bound for halide ratio
        halide_ratio_upper (float): upper bound for halide ratio
    
    Returns:
        compositions (list): list of compositions
    """
    compositions = []
    comp_hashes = []
    hashed_dataset = hash_dataset(dataset)
    halides = [9, 17]
    metals = [39, 13, 22, 21, 31, 49, 40, 12, 30, 32, 57, 58, 41]
    for i in range(n):
        while True:
            comp = []
            total_atoms = np.random.randint(5, max_atoms + 1)
            num_lithium = np.random.randint(max(1, round(total_atoms * li_ratio_lower)), round(total_atoms * li_ratio_upper) + 1)
            comp.extend([3] * num_lithium)
            num_halides = np.random.randint(round(total_atoms * halide_ratio_lower), round(total_atoms * halide_ratio_upper) + 1)
            comp.extend(np.random.choice(halides, num_halides, replace=True))
            if len(comp) >= total_atoms:
                print('Invalid composition, no space for metals: ', comp)
                continue
            temp_metals = []
            while np.unique(comp).size < max_elements and len(comp) < total_atoms:
                temp_metals = np.random.choice(metals, 1, replace=True)
                comp.append(temp_metals[-1])
            if len(comp) < total_atoms:
                comp.extend(np.random.choice(temp_metals, total_atoms - len(comp), replace=True))
            elems = np.unique(comp)
            freq = [comp.count(elem) for elem in elems]
            smact_valid = smact_validity(elems, freq)
            print('SMACT validity: ', smact_valid)
            if not smact_valid:
                print('Invalid composition')
                continue
            comp_hash = hash_structure(comp)
            if comp_hash not in hashed_dataset and comp_hash not in comp_hashes:
                print('Accepted composition', i, ':', comp)
                comp.sort()
                compositions.append(comp)
                comp_hashes.append(comp_hash)
                break
            else:
                print('Invalid compositon, already occurs in dataset')
    return compositions


def sample_random_composition(dataset, n=5):
    """
    Sample n random compositions from the dataset
    Args:
        dataset (dict): dictionary of dataset
        n (int): number of compositions to sample

    Returns:
        dataset_comps (list): list of compositions
    """
    dataset_comps = []
    for data in dataset:
        data['atomic_numbers'].sort()
        dataset_comps.append(data['atomic_numbers'])
    return list(np.random.choice(dataset_comps, n, replace=False))

