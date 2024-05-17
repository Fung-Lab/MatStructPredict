import numpy as np
from ase import Atoms, io
import torch
from torch_geometric.data import Data
from ase.data import chemical_symbols
import pymatgen, pymatgen.io.ase, pymatgen
from pymatgen.core.structure import Element
from ase.data import chemical_symbols
import itertools
from itertools import product, chain
from pymatgen.io.cif import CifWriter

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

# takes a list of similar elements, base_list, and creates all possible permutations with the replacement elements
def generate_replacements(base_list, replacement):
    replacements = product((base_list[0], replacement), repeat=len(base_list))
    
    possibilities = []

    for i in replacements:
        possibilities.append(list(i))

    return possibilities

# for a given atom, returns a list of all possible replacements atoms based on common oxidation states
def generate_replacement_possibilities(atom):
    element_abbreviations = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og"
    ]

    possibilities = []

    for el in element_abbreviations:
        if len(set(Element(atom).common_oxidation_states).intersection(set(Element(el).common_oxidation_states))) > 0:
            possibilities.append(el)

    return possibilities

# recursive substitution method 
def recursively_create_combinations(ls, index=0):
    new_structures = []

    replacement_possibilities = generate_replacement_possibilities(ls[index][0])

    replacements = []

    for atom in replacement_possibilities:
        replacements.append(generate_replacements(ls[index], atom))

    for x in replacements:
        for y in x:
            new_structure = ls.copy()
            new_structure[index] = y
            new_structures.append(new_structure)

    for struct in new_structures.copy():
        if index != len(ls) - 1:
            new_structures.extend(recursively_create_combinations(struct, index + 1))
    
    return new_structures

# recursively finds new structure possibilities based on a list of ase Atoms object templates
def substitution_discovery(templates):
    all_new_structures = []
    for t in templates:
        crystal = io.read(f'/Users/oscarrivera/MatStructPredict_2/data/{t}')
        crystal = crystal.get_chemical_symbols()

        formatted_crystal = []
        temp_ls = [crystal[0]]
        previous_atom = crystal[0]
        for i in crystal[1:]: 
            if i == previous_atom:
                temp_ls.append(i)
            else:
                previous_atom = i
                formatted_crystal.append(temp_ls)
                temp_ls = [i]

        formatted_crystal.append(temp_ls)

        new_structures = recursively_create_combinations(formatted_crystal)

        unique = []
        for e in new_structures:
            if e not in unique:
                unique.append(e)
        
        # check this method
        for n, u in enumerate(unique):
            flattened_u = chain(*u)
            unique[n] = Atoms(flattened_u)
        
        unique = atoms_to_dict(unique)
        all_new_structures.extend(unique)

    return all_new_structures

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

