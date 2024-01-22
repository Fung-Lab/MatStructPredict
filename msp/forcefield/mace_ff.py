from msp.forcefield.base import ForceField
from mace.calculators import mace_mp
from ase import Atoms


class MACE_FF(ForceField):

    def __init__(self):
        """
        Initialize the MACE forcefield model from https://github.com/ACEsuit/mace.
        
        Default installation can be performed with:
        pip install mace-torch
        
        Or from source.
        """
        
                    
    def train(self):
        """
        Train the force field model on the dataset.
        """


    def update(self):
        """
        Update the force field model on the dataset.
        """
            
    def process_data(self):
        """
        Process data for the force field model.
        """


    def _forward(self):
        """
        Calls model directly
        """    
               
        
    def create_ase_calc(self):
        """
        Returns ase calculator
        """
        calculator = mace_mp(model="large", dispersion=False, default_dtype="float32", device='cuda')        
        return calculator
    
    def atoms_to_data(self):
        """
        
        """         

    def data_to_atoms(self):
        """
        
        """      
