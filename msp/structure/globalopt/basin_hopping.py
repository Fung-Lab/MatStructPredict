from msp.structure.optimizer import Optimizer
from ase.optimize import FIRE
from time import time
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from ase import Atom
from ase.constraints import ExpCellFilter


class BasinHoppingBase(Optimizer):
    def __init__(self, name, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=101, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
        """
        super().__init__(name, hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
        self.steps = steps
        self.hops = hops
        self.dr = dr
        self.max_atom_num = max_atom_num
        self.perturbs = []
        self.perturbs.append(self.perturbPos)
        #self.perturbs.append(self.perturbCell)
        #self.perturbs.append(self.perturbAtomicNum)
        #self.perturbs.append(self.addAtom)
        #self.perturbs.append(self.removeAtom)
        #self.perturbs.append(self.swapAtom)
    
    def perturbPos(self, atoms, **kwargs):
        disp = np.random.uniform(-1., 1., (len(atoms), 3)) * self.dr
        atoms.set_positions(atoms.get_positions() + disp)
        

    def perturbCell(self, atoms, **kwargs):
        disp = np.random.uniform(-1., 1., (3, 3)) * self.dr
        atoms.set_cell(atoms.get_cell()[:] + disp)
        

    def perturbAtomicNum(self, atoms, num_atoms_perturb=1, **kwargs):
        atoms_to_perturb = np.random.randint(len(atoms), size=num_atoms_perturb)
        new_atoms = np.random.randint(1, self.max_atom_num, size=num_atoms_perturb)
        atom_list = atoms.get_atomic_numbers()
        atom_list[atoms_to_perturb] = new_atoms
        atoms.set_atomic_numbers(atom_list)

    def addAtom(self, atoms, **kwargs):
        atoms.append(Atom(np.random.randint(1, self.max_atom_num), position=(0, 0, 0)))
        pos = atoms.get_scaled_positions()
        pos[-1] = np.random.uniform(0., 1., (1, 3))
        atoms.set_scaled_positions(pos)
    
    def removeAtom(self, atoms, **kwargs):
        atoms.pop(np.random.randint(len(atoms)))

    def swapAtom(self, atoms, **kwargs):
        nums = atoms.get_atomic_numbers()
        rand_ind = np.random.randint(len(atoms), size=2)
        nums[rand_ind[0]], nums[rand_ind[1]] = nums[rand_ind[1]], nums[rand_ind[0]]
        atoms.set_atomic_numbers(nums)


        
        



class BasinHoppingASE(BasinHoppingBase):

    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=101, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
        """
        super().__init__("BasinHoppingASE", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.calculator = forcefield.create_ase_calc()

    def predict(self, compositions, init_structures=None, cell_relax=True, topk=1, max_atom_num=101, num_atoms_perturb=1):
        """
        Optimizes the composition using the basin hopping optimizer

        Args:
            composition (str): A string representing a chemical composition

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        if init_structures:
            atoms = init_structures
        else:
            atoms = []
            for comp in compositions:
                atoms.append(self.atom_from_comp(comp))
        #atoms = self.atom_from_comp(composition, cell)    
        #atoms.set_calculator(self.calculator)

        min_atoms = deepcopy(atoms)
        curr_atoms = deepcopy(atoms)
        #curr_atoms.set_calculator(self.calculator)
        min_energy = [1e10] * len(min_atoms)        
        
        for index, atom in enumerate(curr_atoms):
            atom.set_calculator(self.calculator)
            if cell_relax:
                atom = ExpCellFilter(atom)           
            min_energy[index] = atom.get_potential_energy(force_consistent=False)            
            for i in range(self.hops):
                oldEnergy = atom.get_potential_energy(force_consistent=False)
                optimizer = FIRE(atom, logfile=None)
                start_time = time()
                optimizer.run(fmax=0.001, steps=self.steps)
                end_time = time()
                num_steps = optimizer.get_number_of_steps()
                time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
                optimizedEnergy = atom.get_potential_energy(force_consistent=False)
                print('HOP', i, 'took', end_time - start_time, 'seconds')
                print('HOP', i, 'old energy', oldEnergy)
                print('HOP', i, 'optimized energy', optimizedEnergy)
                if optimizedEnergy < min_energy[index]:
                    min_atoms[index] = atom.copy()
                    min_energy[index] = optimizedEnergy
                self.perturbs[np.random.randint(len(self.perturbs))](atom, num_atoms_perturb=num_atoms_perturb)
            print('Min energy', min_energy[index])

        return min_atoms
        
        
class BasinHopping(BasinHoppingBase):
    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=101, **kwargs):
        """
        Initialize
        """
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.forcefield = forcefield
    
    def predict(self, compositions, objective_func, init_structures=None, cell_relax=True, topk=1, batch_size=4, log_per=50, lr=.05, num_atoms_perturb=1):
        if init_structures:
            atoms = init_structures
        else:
            atoms = []
            for comp in compositions:
                atoms.append(self.atom_from_comp(comp))
        min_atoms = deepcopy(atoms)
        min_energy = [1e10] * len(min_atoms)
        for i in range(self.hops):
            print("Hop", i)
            newAtoms, newEnergy = self.forcefield.optimize(atoms, self.steps, objective_func, log_per, lr, batch_size=batch_size, cell_relax=cell_relax)
            for j in range(len(newAtoms)):
                if newEnergy[j] < min_energy[j]:
                    print('Atom changed: index ', j)
                    print(min_atoms[j])
                    print(min_energy[j])
                    print(newAtoms[j])
                    print(newEnergy[j])
                    min_energy[j] = newEnergy[j]
                    min_atoms[j] = newAtoms[j].copy()
            atoms = deepcopy(min_atoms)
            for j in range(len(atoms)):
                self.perturbs[np.random.randint(len(self.perturbs))](atoms[j], num_atoms_perturb=num_atoms_perturb)

        return min_atoms

