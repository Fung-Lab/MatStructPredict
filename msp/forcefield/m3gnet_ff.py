from msp.forcefield.base import ForceField
import matgl
from matgl.ext.ase import M3GNetCalculator
from ase import Atoms


class M3GNet_FF(ForceField):

    def __init__(self):
        """
        Initialize the M3GNet forcefield model from https://github.com/materialsvirtuallab/matgl.
        
        Default installation can be performed with:
        pip install matgl
        
        Or from source.
        """
        ###throws error when using default device (GPU); input tensors may not be passed to the right device        
        self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES").to("cpu")        
        
                    
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
        calculator = M3GNetCalculator(self.pot)        
        return calculator
    
    def atoms_to_data(self):
        """
        
        """     

    def data_to_atoms(self):
        """
        
        """      
    
