
def read_dft_config(path):
    """
    Read DFT config file

    Args:
        path (str): path to DFT config file
    Returns:
        dft_config (dict): dictionary of DFT config
    """
    pass

def setup_DFT(dft_config):
    """
    Setup DFT method

    Args:
        dft_config (dict): dictionary of DFT config
    Returns:
        method (object): DFT method
    """
    pass


class Validate:
    def __init__(self, method, local=False):
        self.method = method
        self.local = local

    def __call__(self, atoms):
        """
        Validate a structure with DFT

        Args:
            atoms (ase.Atoms): ASE atoms object representing a structure

        Returns:
            dict: Dictionary of DFT results
        """
        pass