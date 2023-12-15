from msp.forcefield.base import ForceField

class MDL_FF(ForceField):

    def __init__(self, config):
        """
        Initialize the surrogate model.
        """
        pass

    def train(self, dataset):
        """
        Train the surrogate model on the dataset.
        """
        pass

    def update(self, dataset):
        """
        Update the surrogate model on the dataset.
        """
        pass
        
    def create_ase_calc(self):
        """
        Returns ase calculator
        """
        pass