from msp.optimizer.optimizer import Optimizer
from msp.structure.structure_util import atoms_to_dict, dict_to_atoms
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
    def __init__(self, name, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=100, 
                perturbs=['pos', 'cell', 'atomic_num', 'add', 'remove', 'swap'], elems_to_sample=None, **kwargs):
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
        perturbs_dict = {'pos': self.perturbPos, 'cell': self.perturbCell, 'atomic_num': self.perturbAtomicNum, 
                        'add': self.addAtom, 'remove': self.removeAtom, 'swap': self.swapAtom}
        self.perturbs = [perturbs_dict[perturb] for perturb in perturbs]
        if elems_to_sample is None:
            self.elems = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 
                          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
                          48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
                          89, 90, 91, 92, 93, 94]
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

    def change_temp(self, temp, accepts, interval=10, target_ratio=0.5, rate=0.1):
        if len(accepts) % interval == 0 and len(accepts) != 0:
            if sum(accepts[-interval:]) / interval <= target_ratio:
                temp *= 1 + rate
            else:
                temp /= 1 + rate
            temp = max(0.00001, min(temp, 2))

        return temp

    def change_dr(self, accepts, interval=10, target_ratio=0.5, rate=0.1):
        if len(accepts) % interval == 0 and len(accepts) != 0:
            if sum(accepts[-interval:]) / interval <= target_ratio:
                self.dr /= 1 + rate
            else:
                self.dr *= 1 + rate
            self.dr = max(0.1, min(self.dr, 1))

    def accept(self, old_energy, newEnergy, temp):
        return np.random.rand() < np.exp(-(newEnergy - old_energy) / temp)

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
                res[-1].append(
                    {'hop': i, 'init_loss': old_energy, 'loss': optimized_energy, 'perturb': prev_perturb.__name__,
                     'composition': temp.get_atomic_numbers(),
                     'structure': atoms_to_dict([temp], [optimized_energy])[0]})
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
        min_objective_loss = [1e10] * len(min_atoms)
        best_atoms, best_loss = deepcopy(min_atoms), [1e10] * len(min_atoms)
        best_hop = [0] * len(min_atoms)
        prev_perturb = [self.perturbPos] * len(min_atoms)

        accepts = [[] for _ in range(len(new_atoms))]
        accept_rate = [[] for _ in range(len(new_atoms))]
        temps = [[] for _ in range(len(new_atoms))]
        energies = [[] for _ in range(len(new_atoms))]
        step_sizes = []
        temp = [0.0001 for _ in range(len(new_atoms))]

        prev_step_loss = [1e10] * len(min_atoms)

        res = []
        for _ in range(len(min_atoms)):
            res.append([])
        for i in range(self.hops):
            start_time = time()
            new_atoms, obj_loss, energy_loss, novel_loss, soft_sphere_loss = self.forcefield.optimize(new_atoms, self.steps, objective_func, log_per, lr, batch_size=batch_size, cell_relax=cell_relax, optim=self.optimizer)
            self.change_dr(accepts[0], rate=0.1)
            end_time = time()
            for j in range(len(new_atoms)):
                if len(accepts[j]) % 10 == 0:
                    step_sizes.append(self.dr)
                temp[j] = self.change_temp(temp[j], accepts[j], rate=0.1)
                accept = self.accept(prev_step_loss[j], obj_loss[j], temp[j])
                print("Accept:", accept)
                if accept:
                    min_objective_loss[j] = obj_loss[j]
                    min_atoms[j] = new_atoms[j].copy()
                    if min_objective_loss[j] < best_loss[j]:
                        best_loss[j] = min_objective_loss[j]
                        best_atoms[j] = min_atoms[j].copy()
                        best_hop[j] = i
                prev_step_loss[j] = obj_loss[j]
                energies[j].append(obj_loss[j])
                accepts[j].append(accept)
                if len(accepts[j]) % 10 == 0:
                    accept_rate[j].append(sum(accepts[j][-10:]))
                    temps[j].append(temp[j])
                if getattr(objective_func, 'normalize', False):
                    res[j].append({'hop': i, 'objective_loss': obj_loss[j][0], 'energy_loss': energy_loss[j][0], 'novel_loss': novel_loss[j][0], 'soft_sphere_loss': soft_sphere_loss[j][0],
                                'unnormalized_loss' : objective_func.norm_to_raw_loss(energy_loss[j][0], new_atoms[j].get_atomic_numbers()),
                               'perturb': prev_perturb[j].__name__, 'composition': new_atoms[j].get_atomic_numbers(), 
                               'structure': atoms_to_dict([new_atoms[j]], obj_loss[j])[0]})
                else:
                    res[j].append({'hop': i, 'objective_loss': obj_loss[j][0], 'energy_loss': energy_loss[j][0], 'novel_loss': novel_loss[j][0], 'soft_sphere_loss': soft_sphere_loss[j][0],
                               'perturb': prev_perturb[j].__name__, 'composition': new_atoms[j].get_atomic_numbers(), 
                               'structure': atoms_to_dict([new_atoms[j]], obj_loss[j])[0]})
                print("\tStructure: ", j)
                print("\t\tObjective loss: ", res[j][-1]['objective_loss'])
                print("\t\tEnergy loss: ", res[j][-1]['energy_loss'])
                if getattr(objective_func, 'normalize', False):
                    print("\t\tUnnormalized energy loss: ", res[j][-1]['unnormalized_loss'])
                print("\t\tNovel loss: ", res[j][-1]['novel_loss'])
                print("\t\tSoft sphere loss: ", res[j][-1]['soft_sphere_loss'])
                print("\t\tComposition: ", res[j][-1]['composition'])
                print("\t\tperturb: ", res[j][-1]['perturb'])

            print('HOP', i, 'took', end_time - start_time, 'seconds')
            for j in range(len(new_atoms)):
                rand_ind = np.random.randint(len(self.perturbs))
                prev_perturb[j] = self.perturbs[rand_ind]
                self.perturbs[rand_ind](new_atoms[j], num_atoms_perturb=num_atoms_perturb, num_unique=num_unique)
        avg_loss = 0
        for j, hop in enumerate(best_hop):
            print("Structure: ", j)
            print('\tBest hop: ', hop)
            print("\tObjective loss: ", res[j][hop]['objective_loss'])
            print("\tEnergy loss: ", res[j][hop]['energy_loss'])
            if getattr(objective_func, 'normalize', False):
                print("\tUnnormalized energy loss: ", res[j][hop]['unnormalized_loss'])
            print("\tNovel loss: ", res[j][hop]['novel_loss'])
            print("\tSoft sphere loss: ", res[j][hop]['soft_sphere_loss'])
            avg_loss += best_loss[j]
        print('Avg Objective Loss', avg_loss / len(new_atoms))
        min_atoms = atoms_to_dict(best_atoms, min_objective_loss)
        return res, min_atoms, best_hop, energies, accepts, accept_rate, temps, step_sizes

