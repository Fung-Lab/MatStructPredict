from msp.forcefield.base import ForceField

import torch
import yaml
from matdeeplearn.common.registry import registry
from matdeeplearn.common.ase_utils import MDLCalculator


class MDL_FF(ForceField):

    def __init__(self, config):
        """
        Initialize the surrogate model.
        """
        if isinstance(config, str):
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)     
        
        #to be added        
        self.trainer = None
        self.model = self.trainer.model
        
                    
    def train(self, dataset):
        """
        Train the force field model on the dataset.
        """
        
        self.trainer.dataset = dataset
        #initialize new model
        self.trainer.model = trainer._load_model()
        self.trainer.train()
        
        pass

    def update(self, dataset):
        """
        Update the force field model on the dataset.
        """
        self.trainer.dataset = dataset
        self.trainer.train()
                
        pass

    def forward(self, data):
        """
        Calls model directly
        """
        output = self.model(data)
        #output is a dict
        return output
        
    def create_ase_calc(self):
        """
        Returns ase calculator
        """
        calculator = MDLCalculator(config=self.config)        
        return calculator