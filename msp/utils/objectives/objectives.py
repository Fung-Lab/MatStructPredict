import torch

class UpperConfidenceBound(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        """
        Initialize
        """
        pass
        
class Energy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        """
        Initialize
        """
        pass
                
    def forward(self, model_output):    
        return model_output["potential_energy"]
        
class Uncertainty(torch.nn.Module):

    def __init__(self):
        super().__init__()
        """
        Initialize
        """
        pass
                
    def forward(self, model_output):    
        return model_output["potential_energy_uncertainty"]    