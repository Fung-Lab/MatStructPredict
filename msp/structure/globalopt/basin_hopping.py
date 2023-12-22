from msp.structure.optimizer import Optimizer
from ase.optimize import FIRE
from time import time
import numpy as np

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
        super().__init__("BasinHopping", hops=hops, steps=steps, optimizer=optimizer, dr=dr, **kwargs)
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
        atoms = self.atom_from_dict(composition, cell)
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
    def __init__(self, calculator, hops=5, steps=100, optimizer="FIRE", **kwargs):
        """
        Initialize
        """
        pass
    
    def predict(self, composition, topk=1):
        min_atoms=[[]]
        return min_atoms