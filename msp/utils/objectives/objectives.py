import torch
from mp_api.client import MPRester
from ase.data import chemical_symbols

class UpperConfidenceBound(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        """
        Initialize
        """
        pass
        
class Energy(torch.nn.Module):

    def __init__(self, normalize=True):
        super().__init__()
        """
        Initialize
        """
        self.normalize = normalize
        if normalize:
            # mpr = MPRester("tRaolKVwFDg3U7aT3xQVoge59TVW4sk2")
            # element_list = chemical_symbols[1:101]
            # element_energy={}
            # energies = [-1]
            # for element in element_list:
            #     entries = mpr.get_entries_in_chemsys(elements=[element], additional_criteria={"thermo_types": ["GGA_GGA+U"]})
            #     lowest_energy=999
            #     for entry in entries:
            #         if entry.energy_per_atom < lowest_energy:
            #             lowest_energy = entry.energy_per_atom
            #     element_energy[element] = lowest_energy
            #     energies.append(lowest_energy)
            # print(element_energy)
            # print(energies)
            self.element_energy = [-10000, -3.392726045, -0.00905951, -1.9089228666666667, -3.739412865, -6.679391770833334,
                                -9.2286654925, -8.336494925, -4.947961005, -1.9114789675, -0.02593678, -1.3225252934482759, 
                                -1.60028005, -3.74557583, -5.42531803, -5.413302506666667, -4.136449866875, -1.84853666, 
                                -0.06880822, -1.110398947, -2.00559988, -6.332469105, -7.895492016666666, -9.08390607, -9.65304747, 
                                -9.162015292068965, -8.47002121, -7.108317795, -5.78013668, -4.09920667, -1.25974361, -3.0280960225, 
                                -4.623027855, -4.659118405, -3.49591147765625, -1.636946535, -0.05671467, -0.9805340725, 
                                -1.6894934533333332, -6.466471113333333, -8.54770063, -10.10130504, -10.84565011, -10.360638945, 
                                -9.27440254, -7.36430787, -5.17988181, -2.8325560033333335, -0.92288976, -2.75168373, -4.009571855, 
                                -4.12900124, -3.1433058933333338, -1.524012615, -0.03617417, -0.8954023720689656, -1.91897494, 
                                -4.936007105, -5.933089155, -4.780905755, -4.7681474325, -4.7505423225, -4.718586135, -10.2570018, 
                                -14.07612224, -4.6343661, -4.60678684, -4.58240887, -4.56771881, -4.475835423333334, 999, -4.52095052, 
                                -9.95718903, -11.85777763, -12.95813023, -12.444527185, -11.22736743, -8.83843418, -6.07113332, -3.273882, 
                                -0.303680365, -2.3626431466666666, -3.71264707, -3.89003431, -10000, -10000, -10000, -10000, -10000, -4.1211750075, 
                                -7.41385825, -9.51466466, -11.29141001, -12.94777968125, -14.26783833, -10000, -10000, -10000, -10000, -10000, -10000]

    def set_norm_offset(self, z, n_atoms):
        self.offset = [0]*len(n_atoms)
        curr = 0
        for i in range(len(n_atoms)):
            temp = z[curr:curr+n_atoms[i]]
            for j in temp:
                self.offset[i] -= self.element_energy[j]
            curr += n_atoms[i]

    def forward(self, model_output, n_atoms=[]):
        if self.normalize:
            for i in range(len(model_output['potential_energy'])):
                model_output['potential_energy'][i] = (model_output['potential_energy'][i] + self.offset[i]) / n_atoms[i]
            return model_output["potential_energy"]
        else:    
            return model_output["potential_energy"]
    
    def norm_to_raw_loss(self, loss, z):
        offset = 0
        for num in z:
            offset -= self.element_energy[num]
        loss *= len(z)
        loss -= offset
        return loss
        
class Uncertainty(torch.nn.Module):

    def __init__(self):
        super().__init__()
        """
        Initialize
        """
        pass
                
    def forward(self, model_output, **kwargs):    
      
        return model_output["potential_energy_uncertainty"]

class EnergyAndUncertainty(torch.nn.Module):
    def __init__(self, ratio=.5, normalize=True):
        super().__init__()
        """
        Initialize
        """
        self.ratio = ratio
        self.normalize = normalize
        if normalize:
            self.element_energy = [-1, -3.392726045, -0.00905951, -1.9089228666666667, -3.739412865, -6.679391770833334,
                                -9.2286654925, -8.336494925, -4.947961005, -1.9114789675, -0.02593678, -1.3225252934482759, 
                                -1.60028005, -3.74557583, -5.42531803, -5.413302506666667, -4.136449866875, -1.84853666, 
                                -0.06880822, -1.110398947, -2.00559988, -6.332469105, -7.895492016666666, -9.08390607, -9.65304747, 
                                -9.162015292068965, -8.47002121, -7.108317795, -5.78013668, -4.09920667, -1.25974361, -3.0280960225, 
                                -4.623027855, -4.659118405, -3.49591147765625, -1.636946535, -0.05671467, -0.9805340725, 
                                -1.6894934533333332, -6.466471113333333, -8.54770063, -10.10130504, -10.84565011, -10.360638945, 
                                -9.27440254, -7.36430787, -5.17988181, -2.8325560033333335, -0.92288976, -2.75168373, -4.009571855, 
                                -4.12900124, -3.1433058933333338, -1.524012615, -0.03617417, -0.8954023720689656, -1.91897494, 
                                -4.936007105, -5.933089155, -4.780905755, -4.7681474325, -4.7505423225, -4.718586135, -10.2570018, 
                                -14.07612224, -4.6343661, -4.60678684, -4.58240887, -4.56771881, -4.475835423333334, 999, -4.52095052, 
                                -9.95718903, -11.85777763, -12.95813023, -12.444527185, -11.22736743, -8.83843418, -6.07113332, -3.273882, 
                                -0.303680365, -2.3626431466666666, -3.71264707, -3.89003431, 999, 999, 999, 999, 999, -4.1211750075, 
                                -7.41385825, -9.51466466, -11.29141001, -12.94777968125, -14.26783833, 999, 999, 999, 999, 999, 999]

                
    def set_norm_offset(self, z, n_atoms):
        self.offset = [0]*len(n_atoms)
        curr = 0
        for i in range(len(n_atoms)):
            temp = z[curr:curr+n_atoms[i]]
            for j in temp:
                self.offset[i] -= self.element_energy[j]
            curr += n_atoms[i]
        

    def forward(self, model_output, n_atoms=[]):
        if self.normalize:
            for i in range(len(n_atoms)):
                model_output['potential_energy'][i] = (model_output['potential_energy'][i] + self.offset[i]) / n_atoms[i]
            return model_output["potential_energy"] - self.ratio * model_output["potential_energy_uncertainty"]
        else:   
            return model_output["potential_energy"] - self.ratio * model_output["potential_energy_uncertainty"]
