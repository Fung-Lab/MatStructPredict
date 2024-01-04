from msp.structure.optimizer import Optimizer
from ase.optimize import FIRE
from time import time
import numpy as np
from copy import deepcopy

class BasinHoppingASE(Optimizer):

    def __init__(self, calculator, hops=5, steps=100, optimizer="FIRE", dr=.5, **kwargs):
        """
        Initialize the basin hopping optimizer.

        Args:
            calculator: ASE calculator to use for the optimization
            hops (int, optional): Number of basin hops. Defaults to 5.
            steps (int, optional): Number of steps per basin hop. Defaults to 100.
            optimizer (str, optional): Optimizer to use for each step. Defaults to "FIRE".
        """
        super().__init__("BasinHoppingASE", hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
        self.calculator = calculator
        self.hops = hops
        self.steps = steps
        self.dr = dr

    def predict(self, composition, cell=[5, 5, 5, 90, 90, 90], topk=1):
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
            disp = np.random.uniform(-1., 1., (len(atoms), 3)) * self.dr
            curr_atoms.set_positions(curr_atoms.get_positions() + disp)

        return min_atoms
        
        
class BasinHopping(Optimizer):
    def __init__(self, forcefield, hops=5, steps=100, optimizer="FIRE", dr=.5, **kwargs):
        """
        Initialize
        """
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
        self.forcefield = forcefield
        self.hops = hops
        self.steps = steps
        self.dr = dr
    
    def predict(self, compositions, cell=[5, 5, 5, 90, 90, 90], topk=1, batch_size=4, log_per=50, lr=.05):
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
                atoms[j].set_positions(atoms[j].get_positions() + np.random.uniform(-1., 1., (len(atoms[j]), 3)) * self.dr)

        return min_atoms

