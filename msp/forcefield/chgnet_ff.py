from msp.forcefield.base import ForceField
from ase import Atoms


class CHGNet_FF(ForceField):

    def __init__(self):
        """
        Initialize the CHGNet forcefield model from https://github.com/CederGroupHub/chgnet.
        
        Default installation can be performed with:
        pip install chgnet
        
        Or from source.
        
        Recommended to use:
        pip install git+https://github.com/CederGroupHub/chgnet
        """
        ###throws error when using default device (GPU); input tensors may not be passed to the right device
        from chgnet.model.dynamics import CHGNetCalculator
        self.CHGNetCalculator = CHGNetCalculator
        
                    
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
        calculator = self.CHGNetCalculator()        
        return calculator
    
    def atoms_to_data(self):
        """
        
        """     

    def data_to_atoms(self):
        """
        
        """      
    
