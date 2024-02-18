from msp.structure.optimizer import Optimizer
import ase.optimize
from time import time
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from ase import Atom, Atoms
from ase.constraints import ExpCellFilter
import gc
import torch


class BasinHoppingBase(Optimizer):
    def __init__(self, name, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
            dr (int, optional): rate at which to change values
            max_atom_num (int, optional): maximum atom number to be considered
        """
        super().__init__(name, hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
        self.steps = steps
        self.hops = hops
        self.dr = dr
        self.max_atom_num = max_atom_num
        self.perturbs = []
        self.perturbs.append(self.perturbPos)
        self.optimizer = optimizer
        self.perturbs.append(self.perturbCell)
        self.perturbs.append(self.perturbAtomicNum)
        self.perturbs.append(self.addAtom)
        self.perturbs.append(self.removeAtom)
        self.perturbs.append(self.swapAtom)
    
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
        

    def perturbAtomicNum(self, atoms, num_atoms_perturb=1, **kwargs):
        """
        Perturbs the atomic numbers of the atoms in the structure
        """
        if isinstance(atoms, ExpCellFilter):
            atoms_to_perturb = np.random.randint(len(atoms.atoms), size=num_atoms_perturb)
            new_atoms = np.random.randint(1, self.max_atom_num, size=num_atoms_perturb)
            atom_list = atoms.atoms.get_atomic_numbers()
            atom_list[atoms_to_perturb] = new_atoms
            atoms.atoms.set_atomic_numbers(atom_list)
        else:
            atoms_to_perturb = np.random.randint(len(atoms), size=num_atoms_perturb)
            new_atoms = np.random.randint(1, self.max_atom_num, size=num_atoms_perturb)
            atom_list = atoms.get_atomic_numbers()
            atom_list[atoms_to_perturb] = new_atoms
            atoms.set_atomic_numbers(atom_list)

    def addAtom(self, atoms, **kwargs):
        """
        Adds an atom to the structure
        """
        if isinstance(atoms, ExpCellFilter):
            atoms.atoms.append(Atom(np.random.randint(1, self.max_atom_num), position=(0, 0, 0)))
            pos = atoms.atoms.get_scaled_positions()
            pos[-1] = np.random.uniform(0., 1., (1, 3))
            atoms.atoms.set_scaled_positions(pos)
        else:
            atoms.append(Atom(np.random.randint(1, self.max_atom_num), position=(0, 0, 0)))
            pos = atoms.get_scaled_positions()
            pos[-1] = np.random.uniform(0., 1., (1, 3))
            atoms.set_scaled_positions(pos)
    
    def removeAtom(self, atoms, **kwargs):
        """
        Removes an atom from the structure
        """
        if isinstance(atoms, ExpCellFilter):
            atoms.atoms.pop(np.random.randint(len(atoms.atoms)))
        else:
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

    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, **kwargs):
        """
        Initialize the basinhoppingASE optimizer, which uses an ASE calculator to optimize structures one at a time.

        Args:
            forcefield: Takes a forcefield object with a create_ase_calc() function for the caclculator
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
            dr (int, optional): rate at which to change values. Defaults to .5.
            max_atom_num (int, optional): maximum atom number to be considered, exclusive. Defaults to 101.
        """

        super().__init__("BasinHoppingASE", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.calculator = forcefield.create_ase_calc()

    def predict(self, compositions, init_structures=None, cell_relax=True, topk=1, num_atoms_perturb=1, density=.2):
        """
        Optimizes the list of compositions one at a time using the an ASE Calculator

        Args:
            compositions (list): A list of compositions, which are lists of atomic numbers
            init_structures (list, optional): Initialized ase atoms structures to use instead of creating randomized structures. Defaults to None
            cell_relax (bool, optional): whether to relax cell or not. Defaults to True.
            topk (int, optional): Number of best performing structures to save per composition. Defaults to 1.
            num_atoms_perturb (int, optional): number of atoms to perturb for perturbAtomicNum. Defaults to 1.

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        if init_structures:
            atoms = init_structures
        else:
            atoms = []
            for comp in compositions:
                atoms.append(self.atom_from_comp(comp, density))
        #atoms = self.atom_from_comp(composition, cell)    
        #atoms.set_calculator(self.calculator)

        min_atoms = deepcopy(atoms)
        curr_atoms = deepcopy(atoms)
        #curr_atoms.set_calculator(self.calculator)
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
                res[-1].append({'hop': i, 'init_loss': old_energy, 'loss': optimized_energy, 'perturb': prev_perturb.__name__, 'composition': temp.get_atomic_numbers(), 'structure': self.atoms_to_dict([temp], [optimized_energy])[0]})
                prev_perturb = self.perturbs[np.random.randint(len(self.perturbs))]
                prev_perturb(atom, num_atoms_perturb=num_atoms_perturb)
            print('Structure', index, 'Min energy', min_energy[index])
        min_atoms = self.atoms_to_dict(min_atoms, min_energy)
        return res, min_atoms
        
        
class BasinHoppingBatch(BasinHoppingBase):
    def __init__(self, forcefield, hops=5, steps=100, optimizer="Adam", dr=.5, max_atom_num=100, **kwargs):
        """
        Initialize the basinhopping optimizer, which uses a forcefield to optimize batches

        Args:
            forcefield: Takes a forcefield object with a create_ase_calc() function for the caclculator
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "Adam".
            dr (int, optional): rate at which to change values. Defaults to .5.
            max_atom_num (int, optional): maximum atom number to be considered, exclusive. Defaults to 101.
        """
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.forcefield = forcefield
    
    def predict(self, compositions, objective_func, init_structures=None, cell_relax=True, topk=1, batch_size=4, log_per=0, lr=.05, density=.2, num_atoms_perturb=1):
        """
        Optimizes the list of compositions in batches 

        Args:
            compositions (list): A list of compositions, which are lists of atomic numbers
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
        if init_structures:
            atoms = init_structures
        else:
            atoms = []
            for comp in compositions:
                atoms.append(self.atom_from_comp(comp, density))
        min_atoms = deepcopy(atoms)
        min_loss = [1e10] * len(min_atoms)
        best_hop = [0] * len(min_atoms)
        prev_perturb = [self.perturbPos] * len(min_atoms)
        res = []
        for _ in range(len(min_atoms)):
            res.append([])
        for i in range(self.hops):
            start_time = time()
            newAtoms, new_loss, prev_loss = self.forcefield.optimize(atoms, self.steps, objective_func, log_per, lr, batch_size=batch_size, cell_relax=cell_relax, optim=self.optimizer)
            end_time = time()
            print('HOP', i, 'took', end_time - start_time, 'seconds')
            for j in range(len(newAtoms)):
                print('\tStructure', j)
                print('\t\tHOP', i, 'previous energy', prev_loss[j])
                print('\t\tHOP', i, 'optimized energy', new_loss[j])  
                if new_loss[j] < min_loss[j]:
                    min_loss[j] = new_loss[j]
                    min_atoms[j] = newAtoms[j].copy()
                    best_hop[j] = i
                res[j].append({'hop': i, 'init_loss': prev_loss[j][0], 'loss': new_loss[j][0], 'perturb': prev_perturb[j].__name__, 'composition': newAtoms[j].get_atomic_numbers(), 'structure': self.atoms_to_dict([newAtoms[j]], new_loss[j])[0]})
            atoms = deepcopy(min_atoms)
            print('HOP', i, 'took', end_time - start_time, 'seconds')
            for j in range(len(atoms)):
                rand_ind = np.random.randint(len(self.perturbs))
                prev_perturb[j] = self.perturbs[rand_ind]
                self.perturbs[rand_ind](atoms[j], num_atoms_perturb=num_atoms_perturb)
        avg_loss = 0
        for j in range(len(newAtoms)):
            print('Structure', j, 'min energy', min_loss[j], 'best_hop', best_hop[j])
            avg_loss += min_loss[j]
        print('Avg loss', avg_loss / len(newAtoms))
        min_atoms = self.atoms_to_dict(min_atoms, min_loss)
        return res, min_atoms

