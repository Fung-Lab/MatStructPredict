import numpy as np
from ase import Atoms
import torch
from torch_geometric.data import Data
from ase.data import chemical_symbols
import smact
from smact.screening import pauling_test
import itertools



def init_structure(composition, pyxtal=False, density=.2):
    """
    Creates a dictionary representing a structure from a composition
    
    Args:
        composition (list): A list of the atomic numbers
    
    Returns:
        dict: representing structure
    """
    atoms = None
    if not pyxtal:
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
    else:
        from pyxtal import pyxtal
        struc = pyxtal()
        unique_nums = list(set(composition))
        counts = [composition.count(num) for num in unique_nums]
        symbols = [chemical_symbols[num] for num in unique_nums]
        struct_num = 0
        use_random = False
        for i in range(1, 231):
            try:
                use_random = True
                struct_num = i
                struc.from_random(3, i, symbols, counts)
                break
            except:
                continue
        if use_random:
            print('Composition ', composition, 'not compatible with pyxtal. Using random structure')
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
        else:
            print('Using pyxtal group', struct_num)
            atoms = struc.to_ase()
    
    return atoms_to_dict([atoms], [None])[0]

def atoms_to_dict(atoms, loss=None):
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
        if loss is None:
            d['loss'] = None
        else:
            d['loss'] = loss[i]
    return res

def dict_to_atoms(dictionaries):
    """
    Creates ASE atoms objects from a list of dictionaries
    
    Args:
        dictionaries (list): A list of dictionaries representing atoms
    
    Returns:
        list: A list of ASE atoms objects.
    """
    res = []
    for d in dictionaries:
        res.append(Atoms(d['z'], cell=d['cell'], positions=d['pos']))
    return res

def atoms_to_data(atoms):
    """
    Converts a list of ASE atoms objects to a list of torch_geometric.data.Data objects.

    Args:
        atoms (list): A list of ASE atoms objects.
    Returns:
        list: A list of torch_geometric.data.Data objects.
    """
    n_structures = len(atoms)
    data_list = [Data() for _ in range(n_structures)]

    for i, s in enumerate(atoms):
        data = s

        pos = torch.tensor(data.get_positions(), dtype=torch.float)
        cell = torch.tensor(np.array([data.cell[:]]), dtype=torch.float)
        atomic_numbers = torch.LongTensor(data.numbers)
        structure_id = str(i)
                
        data_list[i].n_atoms = len(atomic_numbers)
        data_list[i].pos = pos
        data_list[i].cell = cell   
        data_list[i].structure_id = [structure_id]  
        data_list[i].z = atomic_numbers
        data_list[i].u = torch.Tensor(np.zeros((3))[np.newaxis, ...])

    return data_list

def data_to_atoms(batch):
    """
    Converts a list of torch_geometric.data.Data objects to a list of ASE atoms objects.
    
    Args:
        batch (list): A list of torch_geometric.data.Data objects.
    Returns:
        list: A list of ASE atoms objects.
    """
    res = []
    curr = 0
    for i in range(len(batch.n_atoms)):
        res.append(Atoms(batch.z[curr:curr+batch.n_atoms[i]].cpu().numpy(), cell=batch.cell[i].cpu().detach().numpy(), pbc=(True, True, True), positions=batch.pos[curr:curr+batch.n_atoms[i]].cpu().detach().numpy()))
        curr += batch.n_atoms[i]
    return res

def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False

