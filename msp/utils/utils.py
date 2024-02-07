from ase import Atoms

def atoms_from_dict(dictionaries):
        """
        Creates an ASE atoms object from a composition list
        
        Args:
            composition (list): A list representing the atomic numbers
        
        Returns:
            ase.Atoms: An ASE atoms object representing the composition
        """
        res = []
        for d in dictionaries:
            res.append(Atoms(d['z'], cell=d['cell'], positions=d['pos']))
        return res