from msp.optimizer.optimizer import Optimizer
from msp.structure.structure_util import atoms_to_dict, dict_to_atoms
import ase.optimize
from ase.optimize import FIRE
from time import time
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from ase import Atom, Atoms
from ase.constraints import ExpCellFilter
import gc
import torch


class BasinHoppingBase(Optimizer):
    def __init__(self, name, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, 
                perturbs=['pos', 'cell', 'atomic_num', 'add', 'remove', 'swap'], elems_to_sample=None, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
            dr (int, optional): rate at which to change values.
            max_atom_num (int, optional): maximum atom number to be considered.
            perturbs (list): list of perturbs to use. Defaults to all possible (pos, cell, atomic_num, add, remove, swap).
        """
        super().__init__(name, hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
        self.steps = steps
        self.hops = hops
        self.dr = dr
        self.max_atom_num = max_atom_num
        self.optimizer = optimizer
        perturbs_dict = {'pos': self.perturbPos, 'cell': self.perturbCell, 'atomic_num': self.perturbAtomicNum, 
                        'add': self.addAtom, 'remove': self.removeAtom, 'swap': self.swapAtom}
        self.perturbs = [perturbs_dict[perturb] for perturb in perturbs]
        if elems_to_sample is None:
            self.elems = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 
                          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
                          48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 
                          72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]
        else:
            self.elems = elems_to_sample
    
    def perturbPos(self, atoms, **kwargs):
        """
        Perturbs the positions of the atoms in the structure
        """
        if isinstance(atoms, ExpCellFilter):
            disp = np.random.uniform(-1., 1., (len(atoms.atoms), 3)) * self.dr
            atoms.atoms.set_scaled_positions(atoms.atoms.get_scaled_positions() + disp)
        else:  
            disp = np.random.uniform(-1., 1., (len(atoms), 3)) * self.dr
            atoms.set_scaled_positions(atoms.get_scaled_positions() + disp) 
        

    def perturbCell(self, atoms, **kwargs):
        """
        Perturbs the cell of the atoms in the structure
        """
        if isinstance(atoms, ExpCellFilter):
            disp = np.random.uniform(-1., 1., (3, 3)) * self.dr
            atoms.atoms.set_cell(atoms.get_cell()[:] + disp)
        else:
            disp = np.random.uniform(-1., 1., (3, 3)) * self.dr
            atoms.set_cell(atoms.get_cell()[:] + disp)
        

    def perturbAtomicNum(self, atoms, num_atoms_perturb=1, num_unique=4, **kwargs):
        """
        Perturbs the atomic numbers of the atoms in the structure
        """
        if isinstance(atoms, ExpCellFilter):
            un = np.unique(atoms.atoms.get_atomic_numbers())
            if len(un) < num_unique:
                atoms_to_perturb = np.random.randint(len(atoms.atoms), size=num_atoms_perturb)
                new_atoms = np.random.choice(self.elems, size=num_atoms_perturb)
                atom_list = atoms.atoms.get_atomic_numbers()
                atom_list[atoms_to_perturb] = new_atoms
                atoms.atoms.set_atomic_numbers(atom_list)
            else:
                atoms_to_perturb = np.random.randint(len(atoms.atoms), size=num_atoms_perturb)
                new_atoms = np.random.choice(un, size=num_atoms_perturb)
                atom_list = atoms.atoms.get_atomic_numbers()
                atom_list[atoms_to_perturb] = new_atoms
                atoms.atoms.set_atomic_numbers(atom_list)
        else:
            un = np.unique(atoms.get_atomic_numbers())
            if len(un) < 4:
                atoms_to_perturb = np.random.randint(len(atoms), size=num_atoms_perturb)
                new_atoms = np.random.choice(self.elems, size=num_atoms_perturb)
                atom_list = atoms.get_atomic_numbers()
                atom_list[atoms_to_perturb] = new_atoms
                atoms.set_atomic_numbers(atom_list)
            else:
                atoms_to_perturb = np.random.randint(len(atoms), size=num_atoms_perturb)
                new_atoms = np.random.choice(un, size=num_atoms_perturb)
                atom_list = atoms.get_atomic_numbers()
                atom_list[atoms_to_perturb] = new_atoms
                atoms.set_atomic_numbers(atom_list)

    def addAtom(self, atoms, num_unique=4, **kwargs):
        """
        Adds an atom to the structure
        """
        if isinstance(atoms, ExpCellFilter):
            un = np.unique(atoms.atoms.get_atomic_numbers())
            if len(un) < num_unique:
                atoms.atoms.append(Atom(np.random.choice(self.elems, size=1)[0], position=(0, 0, 0)))
                pos = atoms.atoms.get_scaled_positions()
                pos[-1] = np.random.uniform(0., 1., (1, 3))
                atoms.atoms.set_scaled_positions(pos)
            else:
                atoms.atoms.append(Atom(np.random.choice(un, size=1)[0], position=(0, 0, 0)))
                pos = atoms.atoms.get_scaled_positions()
                pos[-1] = np.random.uniform(0., 1., (1, 3))
                atoms.atoms.set_scaled_positions(pos)
        else:
            un = np.unique(atoms.get_atomic_numbers())
            if len(un) < 4:
                atoms.append(Atom(np.random.choice(self.elems, size=1)[0], position=(0, 0, 0)))
                pos = atoms.get_scaled_positions()
                pos[-1] = np.random.uniform(0., 1., (1, 3))
                atoms.set_scaled_positions(pos)
            else:
                atoms.append(Atom(np.random.choice(un, size=1)[0], position=(0, 0, 0)))
                pos = atoms.get_scaled_positions()
                pos[-1] = np.random.uniform(0., 1., (1, 3))
                atoms.set_scaled_positions(pos)
    
    def removeAtom(self, atoms, **kwargs):
        """
        Removes an atom from the structure
        """
        if isinstance(atoms, ExpCellFilter):
            if len(atoms.atoms) > 2:
                atoms.atoms.pop(np.random.randint(len(atoms.atoms)))
        else:
            if len(atoms) > 2:
                atoms.pop(np.random.randint(len(atoms)))

    def swapAtom(self, atoms, **kwargs):
        """
        Swaps two atoms in the structure
        """
        if isinstance(atoms, ExpCellFilter):
            nums = atoms.atoms.get_atomic_numbers()
            rand_ind = np.random.randint(len(atoms.atoms), size=2)
            nums[rand_ind[0]], nums[rand_ind[1]] = nums[rand_ind[1]], nums[rand_ind[0]]
            atoms.atoms.set_atomic_numbers(nums)
        else:
            nums = atoms.get_atomic_numbers()
            rand_ind = np.random.randint(len(atoms), size=2)
            nums[rand_ind[0]], nums[rand_ind[1]] = nums[rand_ind[1]], nums[rand_ind[0]]
            atoms.set_atomic_numbers(nums)


        
        



class BasinHoppingASE(BasinHoppingBase):

    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, 
                perturbs=['pos', 'cell', 'atomic_num', 'add', 'remove', 'swap'], elems_to_sample=None, **kwargs):
        """
        Initialize the basinhoppingASE optimizer, which uses an ASE calculator to optimize structures one at a time.

        Args:
            forcefield: Takes a forcefield object with a create_ase_calc() function for the caclculator
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
            dr (int, optional): rate at which to change values. Defaults to .5.
            max_atom_num (int, optional): maximum atom number to be considered, exclusive. Defaults to 101.
            perturbs (list): list of perturbs to use. Defaults to all possible (pos, cell, atomic_num, add, remove, swap).
        """

        super().__init__("BasinHoppingASE", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, perturbs=perturbs, elems_to_sample=elems_to_sample, **kwargs)
        self.calculator = forcefield.create_ase_calc()

    def predict(self, structures, cell_relax=True, topk=1, num_atoms_perturb=1, num_unique=4, density=.2):
        """
        Optimizes the list of compositions one at a time using the an ASE Calculator

        Args:
            atoms (list): A list of dictionaries representing atomic structures
            init_structures (list, optional): Initialized ase atoms structures to use instead of creating randomized structures. Defaults to None
            cell_relax (bool, optional): whether to relax cell or not. Defaults to True.
            topk (int, optional): Number of best performing structures to save per composition. Defaults to 1.
            num_atoms_perturb (int, optional): number of atoms to perturb for perturbAtomicNum. Defaults to 1.

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        atoms = dict_to_atoms(structures)
        min_atoms = deepcopy(atoms)
        curr_atoms = deepcopy(atoms)
        min_energy = [1e10] * len(min_atoms)
        res = []
        
        for index, atom in enumerate(curr_atoms):
            atom.set_calculator(self.calculator)
            if cell_relax:
                atom = ExpCellFilter(atom)
            min_energy[index] = atom.get_potential_energy(force_consistent=False)
            prev_perturb = self.perturbPos
            print('Structure', index)
            res.append([])
            for i in range(self.hops):
                old_energy = atom.get_potential_energy(force_consistent=False)
                optimizer = getattr(ase.optimize, self.optimizer, 'FIRE')(atom, logfile=None)
                start_time = time()
                optimizer.run(fmax=0.001, steps=self.steps)
                end_time = time()
                num_steps = optimizer.get_number_of_steps()
                time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
                optimized_energy = atom.get_potential_energy(force_consistent=False)
                print('\tHOP', i, 'took', end_time - start_time, 'seconds')
                print('\tHOP', i, 'previous energy', old_energy)
                print('\tHOP', i, 'optimized energy', optimized_energy)
                if optimized_energy < min_energy[index]:
                    min_atoms[index] = atom.copy()
                    min_energy[index] = optimized_energy
                if isinstance(atom, ExpCellFilter):
                    temp = atom.atoms
                else:
                    temp = atom
                res[-1].append({'hop': i, 'init_loss': old_energy, 'loss': optimized_energy, 'perturb': prev_perturb.__name__, 'composition': temp.get_atomic_numbers(), 'structure': atoms_to_dict([temp], [optimized_energy])[0]})
                prev_perturb = self.perturbs[np.random.randint(len(self.perturbs))]
                prev_perturb(atom, num_atoms_perturb=num_atoms_perturb, num_unique=num_unique)
            print('Structure', index, 'Min energy', min_energy[index])
        min_atoms = atoms_to_dict(min_atoms, min_energy)
        return res, min_atoms
        
        
class BasinHoppingBatch(BasinHoppingBase):
    def __init__(self, forcefield, hops=5, steps=100, optimizer="Adam", dr=.5, max_atom_num=100,
                perturbs=['pos', 'cell', 'atomic_num', 'add', 'remove', 'swap'], elems_to_sample=None, **kwargs):
        """
        Initialize the basinhopping optimizer, which uses a forcefield to optimize batches

        Args:
            forcefield: Takes a forcefield object with a create_ase_calc() function for the caclculator
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "Adam".
            dr (int, optional): rate at which to change values. Defaults to .5.
            max_atom_num (int, optional): maximum atom number to be considered, exclusive. Defaults to 101.
            perturbs (list): list of perturbs to use. Defaults to all possible (pos, cell, atomic_num, add, remove, swap).
        """
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, perturbs=perturbs, elems_to_sample=elems_to_sample, **kwargs)
        self.forcefield = forcefield
    
    def predict(self, structures, objective_func, cell_relax=True, topk=1, batch_size=4, log_per=0, lr=.05, density=.2, num_atoms_perturb=1, num_unique=4):
        """
        Optimizes the list of compositions in batches 

        Args:
            atoms (list): A list of dictionaries representing atomic structures
            objective_func (func): An evaluation method to compare structures on some basis
            init_structures (list, optional): Initialized ase atoms structures to use instead of creating randomized structures. Defaults to None
            cell_relax (bool, optional): whether to relax cell or not. Defaults to True.
            topk (int, optional): Number of best performing structures to save per composition. Defaults to 1.
            batch_size (int, optional): Batch_size for optimization. Deafults to 4
            log_per (int, optional): Print log messages for every log_per steps. Defaults to 0 (no logging).
            lr (int, optional): Learning rate for optimizer. Defaults to .5.
            num_atoms_perturb (int, optional): number of atoms to perturb for perturbAtomicNum

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        new_atoms = dict_to_atoms(structures)
        min_atoms = deepcopy(new_atoms)
        min_loss = [1e10] * len(min_atoms)
        best_hop = [0] * len(min_atoms)
        prev_perturb = [self.perturbPos] * len(min_atoms)
        res = []
        for _ in range(len(min_atoms)):
            res.append([])
        for i in range(self.hops):
            start_time = time()
            new_atoms, new_loss, prev_loss = self.forcefield.optimize(new_atoms, self.steps, objective_func, log_per, lr, batch_size=batch_size, cell_relax=cell_relax, optim=self.optimizer)
            end_time = time()
            for j in range(len(new_atoms)):
                # print('\tStructure', j)
                # print('\t\tHOP', i, 'previous energy', prev_loss[j])
                # print('\t\tHOP', i, 'optimized energy', new_loss[j])
                # print('\t\tHOP', i, 'raw energy', objective_func.norm_to_raw_loss(new_loss[j][0], new_atoms[j].get_atomic_numbers())) 
                if new_loss[j] < min_loss[j]:
                    min_loss[j] = new_loss[j]
                    min_atoms[j] = new_atoms[j].copy()
                    best_hop[j] = i
                if getattr(objective_func, 'normalize', False):
                    res[j].append({'hop': i, 'init_loss': prev_loss[j][0], 'loss': new_loss[j][0], 'raw_loss' : objective_func.norm_to_raw_loss(new_loss[j][0], new_atoms[j].get_atomic_numbers()),
                               'perturb': prev_perturb[j].__name__, 'composition': new_atoms[j].get_atomic_numbers(), 
                               'structure': atoms_to_dict([new_atoms[j]], new_loss[j])[0]})
                else:
                    res[j].append({'hop': i, 'init_loss': prev_loss[j][0], 'loss': new_loss[j][0],
                               'perturb': prev_perturb[j].__name__, 'composition': new_atoms[j].get_atomic_numbers(), 
                               'structure': atoms_to_dict([new_atoms[j]], new_loss[j])[0]})
                print("\tStructure: ", j)
                print("\t\tInit loss: ", res[j][-1]['init_loss'])
                print("\t\tFinal loss: ", res[j][-1]['loss'])
                if getattr(objective_func, 'normalize', False):
                    print("\t\tRaw loss: ", res[j][-1]['raw_loss'])
                print("\t\tComposition: ", res[j][-1]['composition'])
                print("\t\tperturb: ", res[j][-1]['perturb'])
            print('HOP', i, 'took', end_time - start_time, 'seconds')
            for j in range(len(new_atoms)):
                rand_ind = np.random.randint(len(self.perturbs))
                prev_perturb[j] = self.perturbs[rand_ind]
                self.perturbs[rand_ind](new_atoms[j], num_atoms_perturb=num_atoms_perturb, num_unique=num_unique)
        avg_loss = 0
        for j in range(len(min_loss)):
            print('Structure', j, 'min energy', min_loss[j], 'best_hop', best_hop[j])
            avg_loss += min_loss[j]
        print('Avg loss', avg_loss / len(new_atoms))
        min_atoms = atoms_to_dict(min_atoms, min_loss)
        return res, min_atoms, best_hop

class BasinHoppingCatalyst:
    def __init__(self, forcefield, hops=5, steps=100, optimizer="Adam", dr=.5, max_atom_num=100, 
                 catalyst_elem=78, radius=5.0, **kwargs):
        self.forcefield = forcefield
        self.hops = hops
        self.steps = steps
        self.optimizer = optimizer
        self.dr = dr
        self.max_atom_num = max_atom_num
        self.catalyst_elem = catalyst_elem
        self.radius = radius
    
    def addAtom2(self, atoms, **kwargs):
        print("attempting to add")
        print("atoms: ", atoms)
        if len(atoms) < self.max_atom_num:
            print("adding ")
            pos = np.random.uniform(-self.radius, self.radius, size=(1, 3))
            pos[:, 2] = atoms.positions[:, 2].max()  # Ensure the new atom is on the surface
            atoms.append(Atom(self.catalyst_elem, position=pos[0]))

    def get_surface_atoms_indices(self, atoms):
        positions = [atom.position for atom in atoms]
        x_min, x_max = min(position[0] for position in positions), max(position[0] for position in positions)
        y_min, y_max = min(position[1] for position in positions), max(position[1] for position in positions)
        z_min, z_max = min(position[2] for position in positions), max(position[2] for position in positions)
        threshold = 0.1   # set a cutoff where all atoms greater than cutoff are considered surface
        surface_atoms_indices = [index for index, atom in enumerate(atoms)
                                if (x_max - atom.position[0] <= threshold) or (atom.position[0] - x_min <= threshold) or
                                    (y_max - atom.position[1] <= threshold) or (atom.position[1] - y_min <= threshold) or
                                    (z_max - atom.position[2] <= threshold) or (atom.position[2] - z_min <= threshold)]
        return surface_atoms_indices

    def addAtom(self, atoms, **kwargs):
        if len(atoms) < self.max_atom_num:
            x_range = (min(atom.position[0] for atom in atoms), max(atom.position[0] for atom in atoms))
            y_range = (min(atom.position[1] for atom in atoms), max(atom.position[1] for atom in atoms))
            z_max = max(atom.position[2] for atom in atoms)
            x_pos = np.random.uniform(*x_range)
            y_pos = np.random.uniform(*y_range)

            surface_indices = self.get_surface_atoms_indices(atoms)
            surface_atoms = [atoms[i] for i in surface_indices]

            threshold = 0.1
            z_pos = z_max + threshold

            found_position = False
            while z_pos > 0 and not found_position: 
                for atom in surface_atoms:
                    distance = np.linalg.norm(np.array([x_pos, y_pos, z_pos]) - atom.position)
                    if distance < threshold:  # Threshold for being "close"
                        found_position = True
                        break
                if not found_position:
                    z_pos -= 0.1

            new_atom = Atom(self.catalyst_elem, position=(x_pos, y_pos, z_pos))
            print("Adding atom", new_atom, "at position", (x_pos, y_pos, z_pos))
            atoms.append(new_atom)
        else:
            print("Maximum number of atoms reached. Skipping atom addition.")

    def removeAtom(self, atoms, **kwargs):
        catalyst_indices = [i for i, atom in enumerate(atoms) if atom.number == self.catalyst_elem]
        print("Catalyst indices are:", catalyst_indices)
        if catalyst_indices:
            idx = np.random.choice(catalyst_indices)
            del atoms[idx]
            print("Removed atom at index", idx)
        else:
            print("No catalyst atoms found to remove")
    def perturbPos(self, atoms, **kwargs):
        print("attempting to perturb")
        print("atoms: ", atoms)
        print("catalyst elem is ", self.catalyst_elem)

        catalyst_indices = [i for i, atom in enumerate(atoms) if atom.number == self.catalyst_elem]
        print("catalyst indices are : ", catalyst_indices)
        if catalyst_indices:
            print("perturbing")
            
            idx = np.random.choice(catalyst_indices)
            pos = atoms.positions[idx]
            print("old position is ", pos)
            pos += np.random.uniform(-self.dr, self.dr, size=3)  # Change size to (3,)
            pos[2] = atoms.positions[:, 2].max()  # Ensure the perturbed atom stays on the surface
            atoms.positions[idx] = pos
            print("new position is ", pos)

    def predict(self, structures, objective_func, cell_relax=True, topk=1, batch_size=4, log_per=0, lr=.05, density=.2):
        new_atoms = deepcopy(structures)
        min_atoms = deepcopy(new_atoms)
        min_loss = [1e10] * len(min_atoms)
        best_hop = [0] * len(min_atoms)
        res = []
        for _ in range(len(min_atoms)):
            res.append([])
        for i in range(self.hops):
            start_time = time()
            new_atoms, new_loss, prev_loss = self.forcefield.optimize(new_atoms, self.steps, objective_func, log_per, lr, batch_size=batch_size, cell_relax=cell_relax, optim=self.optimizer)
            end_time = time()
            for j in range(len(new_atoms)):
                print(f"Hop {i}, Structure {j}: {new_atoms[j].get_chemical_formula()}")  # Debugging print statement
                if new_loss[j] < min_loss[j]:
                    min_loss[j] = new_loss[j]
                    min_atoms[j] = new_atoms[j].copy()  # Update min_atoms with the best structure
                    best_hop[j] = i
                if getattr(objective_func, 'normalize', False):
                    res[j].append({'hop': i, 'init_loss': prev_loss[j][0], 'loss': new_loss[j][0], 'raw_loss': objective_func.norm_to_raw_loss(new_loss[j][0], new_atoms[j].get_atomic_numbers()),
                                'composition': new_atoms[j].get_atomic_numbers(), 
                                'structure': atoms_to_dict([new_atoms[j]], new_loss[j])[0]})
                else:
                    res[j].append({'hop': i, 'init_loss': prev_loss[j][0], 'loss': new_loss[j][0],
                                'composition': new_atoms[j].get_atomic_numbers(), 
                                'structure': atoms_to_dict([new_atoms[j]], new_loss[j])[0]})
            print('HOP', i, 'took', end_time - start_time, 'seconds')
            for j in range(len(new_atoms)):
                action = np.random.choice(['add', 'remove', 'perturb'])
                action='perturb'
                if action == 'add':
                    self.addAtom(new_atoms[j])
                elif action == 'remove':
                    self.removeAtom(new_atoms[j])
                else:  # perturb
                    self.perturbPos(new_atoms[j])
        avg_loss = 0
        for j in range(len(min_loss)):
            print('Structure', j, 'min energy', min_loss[j], 'best_hop', best_hop[j])
            print(f"Final Structure {j}: {min_atoms[j].get_chemical_formula()}")  # Debugging print statement
            avg_loss += min_loss[j]
        print('Avg loss', avg_loss / len(new_atoms))
        return res, min_atoms, best_hop

class BasinHoppingSurface(BasinHoppingBase):
    def __init__(self, calculator, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, **kwargs):        
        super().__init__("BasinHoppingSurface", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.calculator=calculator
        self.virtual_sites=[]

    def get_surface_atoms_indices(self, atoms):
        positions = [atom.position for atom in atoms]
        x_min, x_max = min(position[0] for position in positions), max(position[0] for position in positions)
        y_min, y_max = min(position[1] for position in positions), max(position[1] for position in positions)
        z_min, z_max = min(position[2] for position in positions), max(position[2] for position in positions)
        threshold = 1.0   # set a cutoff where all atoms greater than cutoff are considered surface
        surface_atoms_indices = [index for index, atom in enumerate(atoms)
                                if (x_max - atom.position[0] <= threshold) or (atom.position[0] - x_min <= threshold) or
                                    (y_max - atom.position[1] <= threshold) or (atom.position[1] - y_min <= threshold) or
                                    (z_max - atom.position[2] <= threshold) or (atom.position[2] - z_min <= threshold)]
        return surface_atoms_indices
    
    def perturb_surface(self, atoms):
        print("perturbing surface")
        surface_indices = self.get_surface_atoms_indices(atoms)
        if len(surface_indices) == 0:
            return  # No surface atoms to perturb
        move_type = np.random.choice(["add", "replace", "remove"])
        if move_type == "add":
            print("attempting to add ")
            #assert(False)
            #self.addAtomScan(atoms)
            self.add_atom_to_virtual_site(atoms)
            #self.addAtom(atoms)
        elif move_type == "replace":
            print("attempting to replcae")
            self.replaceAtom(atoms, surface_indices)
        elif move_type == "remove":
            print("attempting to remove")
            self.removeAtom(atoms)

    def remove_surface_layers(self, atoms, n_layers): # for virtual adsorption sites
        z_coords = atoms.positions[:, 2] 
        unique_z = np.unique(z_coords)
        
        if len(unique_z) < n_layers:
            raise ValueError("Not enough layers in the structure to remove")

        cutoff_z = sorted(unique_z)[n_layers - 1]
        self.virtual_sites = [atom.position for atom in atoms if atom.position[2] >= cutoff_z]
        del_atoms_indices = [i for i, atom in enumerate(atoms) if atom.position[2] >= cutoff_z]
        del atoms[del_atoms_indices]

    def add_atom_to_virtual_site(self, atoms):
        if not self.virtual_sites:
            raise ValueError("No virtual sites available for adding an atom.")
        unique_atomic_numbers = np.unique(atoms.get_atomic_numbers())
        new_atom_atomic_number = np.random.choice(unique_atomic_numbers)
        site_index = np.random.randint(len(self.virtual_sites))
        new_atom_position = self.virtual_sites[site_index]
        atoms += Atom(new_atom_atomic_number, position=new_atom_position)

    def addAtomScan(self, atoms):
        x_range = (min(atom.position[0] for atom in atoms), max(atom.position[0] for atom in atoms))
        y_range = (min(atom.position[1] for atom in atoms), max(atom.position[1] for atom in atoms))
        z_max = max(atom.position[2] for atom in atoms)
        x_pos = np.random.uniform(*x_range)
        y_pos = np.random.uniform(*y_range)

        surface_indices = self.get_surface_atoms_indices(atoms)
        surface_atoms = [atoms[i] for i in surface_indices]

        threshold = 1.0
        z_pos = z_max + threshold

        found_position = False
        while z_pos > 0 and not found_position:  # Ensure z_pos is always above 0
            for atom in surface_atoms:
                distance = np.linalg.norm(np.array([x_pos, y_pos, z_pos]) - atom.position)
                if distance < threshold:  # Threshold for being "close"
                    found_position = True
                    break
            if not found_position:
                z_pos -= 0.1
        unique_atomic_numbers = np.unique(atoms.get_atomic_numbers())
        new_atom_type = np.random.choice(unique_atomic_numbers)
        new_atom = Atom(new_atom_type, position=(x_pos, y_pos, z_pos))
        print("adding atom ", new_atom)
        atoms.append(new_atom)

    def addAtom(self, atoms):
        surface_indices = self.get_surface_atoms_indices(atoms)
        #assert(False)
        new_atom = Atom(np.random.randint(1, self.max_atom_num), position=(0, 0, 0))
        print("added atom ", new_atom)
        atoms.append(new_atom)
        pos = atoms.get_scaled_positions()
        pos[-1] = np.random.uniform(0., 1., (1, 3))
        atoms.set_scaled_positions(pos)

    def removeAtom(self, atoms):
        surface_indices = self.get_surface_atoms_indices(atoms)
        if len(surface_indices) > 0:
            atom_to_remove = np.random.choice(surface_indices)
            #print("removed1 ", atoms[atom_to_remove])
            del atoms[atom_to_remove]

    def replaceAtom(self, atoms, surface_indices):
        if len(surface_indices) > 0:
            unique_atomic_numbers = np.unique(atoms.get_atomic_numbers())
            index_to_replace = np.random.choice(surface_indices)
            old_atomic_number = atoms[index_to_replace].number
            possible_replacements = unique_atomic_numbers[unique_atomic_numbers != old_atomic_number]
            if len(possible_replacements) > 0:
                new_atomic_number = np.random.choice(possible_replacements)
            else:
                new_atomic_number = old_atomic_number
            
            new_atom = Atom(new_atomic_number, position=atoms[index_to_replace].position)
            print("replaced with ", new_atomic_number," : ", new_atom)
            del atoms[index_to_replace]
            atoms.append(new_atom)
            #atoms[index_to_replace] = new_atom
            # only replace with atoms already in teh set
            # atoms.getatomicnumbers and pick from that list
    
    def predict(self, atoms):
    #def predict(self, composition, cell=np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]), topk=1, max_atom_num=101, num_atoms_perturb=1):
        
        
        min_atoms = atoms.copy()
        min_atoms.set_calculator(self.calculator)

        print("min atoms calculator: ", min_atoms.get_calculator())
        curr_atoms = atoms.copy()
        curr_atoms.set_calculator(self.calculator)
        
        self.remove_surface_layers(curr_atoms, 2)

        min_energy = curr_atoms.get_potential_energy()
        print("minenergy is ", min_energy)
        print("self.hops is ", self.hops)
        # curr_atoms is not a pointer to atoms - fix
        # should be perturbing curr_atoms
        for i in range(self.hops):
            old_energy = curr_atoms.get_potential_energy()
            optimizer = FIRE(curr_atoms, logfile=None)
            optimizer.run(fmax=0.001, steps=self.steps)
            optimized_energy = curr_atoms.get_potential_energy()

            if optimized_energy < min_energy:
                min_atoms = curr_atoms.copy()
                min_energy = optimized_energy
            #self.perturbs[np.random.randint(len(self.perturbs))](curr_atoms)
            self.perturb_surface(curr_atoms)
        return min_atoms