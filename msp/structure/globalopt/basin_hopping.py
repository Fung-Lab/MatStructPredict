from msp.structure.optimizer import Optimizer
from ase.optimize import FIRE
from time import time
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

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
    
    def perturbPos(self, atoms, **kwargs):
        disp = np.random.uniform(-1., 1., (len(atoms), 3)) * self.dr
        atoms.set_positions(atoms.get_positions() + disp)
        

    def perturbCell(self, atoms, **kwargs):
        disp = np.random.uniform(-1., 1., (3, 3)) * self.dr
        atoms.set_cell(atoms.get_cell()[:] + disp)
        

    def perturbAtomicNum(self, atoms, num_atoms_perturb=1):
        atoms_to_perturb = np.random.randint(len(atoms), size=num_atoms_perturb)
        new_atoms = np.random.randint(1, self.max_atom_num, size=num_atoms_perturb)
        atom_list = atoms.get_atomic_numbers()
        atom_list[atoms_to_perturb] = new_atoms
        atom_list.sort()
        atoms.set_atomic_numbers(atom_list)
        
        



class BasinHoppingASE(BasinHoppingBase):

    def __init__(self, calculator, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=101, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
        """
        super().__init__("BasinHoppingASE", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.calculator = calculator

    def predict(self, composition, cell=np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]), topk=1, perturbPos=True, perturbCell=False, perturbAtomicNum=False, max_atom_num=101, num_atoms_perturb=1):
        """
        Optimizes the composition using the basin hopping optimizer

        Args:
            composition (str): A string representing a chemical composition

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        atoms = self.atom_from_comp(composition, cell)
        atoms.set_calculator(self.calculator)

        min_atoms = atoms.copy()
        curr_atoms = atoms.copy()
        curr_atoms.set_calculator(self.calculator)
        min_energy = curr_atoms.get_potential_energy()

        perturbs = []
        if perturbPos:
            pertubs.append(self.perturbPos)
        if perturbCell:
            pertubs.append(self.perturbCell)
        if perturbAtomicNum:
            pertubs.append(self.perturbAtomicNum)

        for i in range(self.hops):
            oldEnergy = curr_atoms.get_potential_energy()
            optimizer = FIRE(curr_atoms, logfile=None)
            start_time = time()
            optimizer.run(fmax=0.001, steps=self.steps)
            end_time = time()
            num_steps = optimizer.get_number_of_steps()
            time_per_step = (end_time - start_time) / num_steps if num_steps != 0 else 0
            optimizedEnergy = curr_atoms.get_potential_energy()
            print('HOP', i, 'took', end_time - start_time, 'seconds')
            print('HOP', i, 'old energy', oldEnergy)
            print('HOP', i, 'optimized energy', optimizedEnergy)
            if optimizedEnergy < min_energy:
                min_atoms = curr_atoms.copy()
                min_energy = optimizedEnergy
            for perturb in perturbs:
                curr_atoms = perturb(curr_atoms, num_atoms_perturb=num_atoms_perturb)

        return min_atoms
        
        
class BasinHopping(BasinHoppingBase):
    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, max_atom_num=101, **kwargs):
        """
        Initialize
        """
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, max_atom_num=max_atom_num, **kwargs)
        self.forcefield = forcefield
    
    def predict(self, compositions, cell=[5, 5, 5, 90, 90, 90], topk=1, batch_size=4, log_per=50, lr=.05,  perturbPos=True, perturbCell=False, perturbAtomicNum=False, num_atoms_perturb=1):

        perturbs = []
        if perturbPos:
            perturbs.append(self.perturbPos)
        if perturbCell:
            perturbs.append(self.perturbCell)
        if perturbAtomicNum:
            perturbs.append(self.perturbAtomicNum)

        atoms = []
        for comp in compositions:
            atoms.append(self.atom_from_comp(comp, cell))
        min_atoms = deepcopy(atoms)
        oldEnergy = [1e10] * len(min_atoms)
        for i in range(self.hops):
            print("Hop", i)
            newAtoms, newEnergy = self.forcefield.optimize(atoms, self.steps, log_per, lr, batch_size=batch_size)
            for j in range(len(newAtoms)):
                if newEnergy[j] < oldEnergy[j]:
                    print('Atom changed: index ', j)
                    print(min_atoms[j])
                    print(oldEnergy[j])
                    print(newAtoms[j])
                    print(newEnergy[j])
                    oldEnergy[j] = newEnergy[j]
                    min_atoms[j] = newAtoms[j].copy()
                atoms = deepcopy(min_atoms)
                for perturb in perturbs:
                    perturb(atoms[j], num_atoms_perturb=num_atoms_perturb)

        return min_atoms

