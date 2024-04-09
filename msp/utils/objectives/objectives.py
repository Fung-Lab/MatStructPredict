import torch
from pathlib import Path
import numpy as np
import time as time
from torch_scatter import scatter_add

        
class Energy(torch.nn.Module):

    def __init__(self, normalize=True, energy_ratio=1.0, ljr_ratio=1.0, ljr_power=12, ljr_scale = .8):
        super().__init__()
        """
        Initialize objective function using only energy and no novel loss
        Args:
            normalize (bool): Whether to normalize the energy by the number of atoms
            energy_ratio (float): Weight of the energy in the loss
            ljr_ratio (float): Weight of the Lennard-Jones repulsion in the loss
            ljr_power (int): Power for the Lennard-Jones repulsion calculation
            ljr_scale (float): Scaling factor for the Lennard-Jones repulsion
        """
        self.normalize = normalize
        self.ljr_power = ljr_power
        self.lj_rmins = np.load(str(Path(__file__).parent / "lj_rmins.npy")) * ljr_scale
        self.ljr_ratio = ljr_ratio
        self.energy_ratio = energy_ratio
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
        """
        Set the offset for the energy normalization
        Args:
            z (torch.Tensor): Atomic numbers of the atoms in the batch
            n_atoms (torch.Tensor): Number of atoms in each structure in the batch
        """
        self.offset = torch.zeros((len(n_atoms), 1)).to(z.device)
        curr = 0
        self.lj_rmins = torch.tensor(self.lj_rmins).to(z.device)
        for i in range(len(n_atoms)):
            temp = z[curr:curr+n_atoms[i]].long()
            for j in temp:
                self.offset[i] -= self.element_energy[j]
            curr += n_atoms[i]
    
    def lj_repulsion(self, data, power = 12):
        """
        Calculate the Lennard-Jones repulsion
        Args:
            data (torch_geometric.data.Data): Data object containing the structure
            power (int): Power for the Lennard-Jones repulsion calculation
        Returns:
            torch.Tensor: Lennard-Jones repulsion
        """
        rmins = self.lj_rmins[(data.z[data.edge_index[0]] - 1), 
            (data.z[data.edge_index[1]] - 1)]
        repulsions = torch.where(rmins <= data.edge_weight, 
            1.0, torch.pow(rmins / data.edge_weight, power))
        edge_idx_to_graph = data.batch[data.edge_index[0]]
        lennard_jones_out = scatter_add(repulsions - 1, index=edge_idx_to_graph, dim_size=len(data))        
        return lennard_jones_out.unsqueeze(1)
    
    def norm_to_raw_loss(self, loss, z):
        """
        Convert normalized loss to raw loss
        Args:
            loss (torch.Tensor): Normalized loss
            z (torch.Tensor): Atomic numbers of the atoms in the batch
        Returns:
            torch.Tensor: Raw loss
        """
        offset = 0
        for num in z:
            offset -= self.element_energy[num]
        loss *= len(z)
        loss -= offset
        return loss

    def forward(self, model_output, batch):
        """
        Forward pass of the objective function
        Args:
            model_output (dict): Output of the model
            batch (torch_geometric.data.Batch): Batch of data
        Returns:
            torch.Tensor: Objective Loss
            torch.Tensor: Potential energy
            torch.Tensor: Novel loss
            torch.Tensor: Lennard-Jones repulsion
        """
        if self.normalize:
            model_output['potential_energy'] = (model_output['potential_energy'] + self.offset) / batch.n_atoms.unsqueeze(1)
            # for i in range(len(batch.n_atoms)):
            #     model_output['potential_energy'][i] = (model_output['potential_energy'][i] + self.offset[i]) / batch.n_atoms[i]
        ljr = self.lj_repulsion(batch, power=self.ljr_power)
        return self.energy_ratio * model_output["potential_energy"] + self.ljr_ratio * ljr, model_output["potential_energy"], torch.zeros(len(model_output['potential_energy']), 1).to(ljr.device), ljr

class EnergyAndUncertainty(Energy):
    def __init__(self, normalize=True, energy_ratio=1.0, ljr_ratio=1, ljr_power=12, ljr_scale=.8, uncertainty_ratio=.25):
        """
        Initialize objective function using energy and uncertainty as novel loss
        Args:
            normalize (bool): Whether to normalize the energy by the number of atoms
            energy_ratio (float): Weight of the energy in the loss
            ljr_ratio (float): Weight of the Lennard-Jones repulsion in the loss
            ljr_power (int): Power for the Lennard-Jones repulsion calculation
            ljr_scale (float): Scaling factor for the Lennard-Jones repulsion
            uncertainty_ratio (float): Weight of the uncertainty in the loss
        """
        super().__init__(normalize, energy_ratio, ljr_ratio, ljr_power, ljr_scale)
        self.uncertainty_ratio = uncertainty_ratio
    
    def forward(self, model_output, batch):
        """
        Forward pass of the objective function
        Args:
            model_output (dict): Output of the model
            batch (torch_geometric.data.Batch): Batch of data
        Returns:
            torch.Tensor: Objective Loss
            torch.Tensor: Potential energy
            torch.Tensor: Novel loss
            torch.Tensor: Lennard-Jones repulsion
        """
        if self.normalize:
            model_output['potential_energy'] = (model_output['potential_energy'] + self.offset) / batch.n_atoms.unsqueeze(1)
            # for i in range(len(batch.n_atoms)):
            #     model_output['potential_energy'][i] = (model_output['potential_energy'][i] + self.offset[i]) / batch.n_atoms[i]
        ljr = self.lj_repulsion(batch, power=self.ljr_power)
        return self.energy_ratio * model_output["potential_energy"] - self.uncertainty_ratio * model_output["potential_energy_uncertainty"] + self.ljr_ratio * ljr, model_output["potential_energy"], -model_output["potential_energy_uncertainty"], ljr




class EmbeddingDistance(Energy):
    def __init__(self, embeddings, normalize=True, energy_ratio=1.0, ljr_ratio=1, ljr_power=12, ljr_scale=.8, embedding_ratio=.1, mode="min"):
        """
        Initialize objective function using only energy and embedding distance as novel loss
            embedding distance is aggregated euclidean distance between structure embedding and database embeddings
        Args:
            embeddings (torch.Tensor): Embeddings of the database structures
            normalize (bool): Whether to normalize the energy by the number of atoms
            energy_ratio (float): Weight of the energy in the loss
            ljr_ratio (float): Weight of the Lennard-Jones repulsion in the loss
            ljr_power (int): Power for the Lennard-Jones repulsion calculation
            ljr_scale (float): Scaling factor for the Lennard-Jones repulsion
            embedding_ratio (float): Weight of the embedding distance in the loss
            mode (str): Aggregation mode for the embedding distance, either "min" or "mean"
        """
        super().__init__(normalize, energy_ratio, ljr_ratio, ljr_power, ljr_scale)
        self.embedding_ratio = embedding_ratio
        self.embeddings = embeddings
        self.mode = mode
                
    def set_norm_offset(self, z, n_atoms):
        """
        Set the offset for the energy normalization
        Args:
            z (torch.Tensor): Atomic numbers of the atoms in the batch
            n_atoms (torch.Tensor): Number of atoms in each structure in the batch
        """
        self.offset = torch.zeros((len(n_atoms), 1)).to(z.device)
        self.embeddings = self.embeddings.to(z.device)
        curr = 0
        self.lj_rmins = torch.tensor(self.lj_rmins).to(z.device)
        for i in range(len(n_atoms)):
            temp = z[curr:curr+n_atoms[i]].long()
            for j in temp:
                self.offset[i] -= self.element_energy[j]
            curr += n_atoms[i]
    
    def forward(self, model_output, batch):
        """
        Forward pass of the objective function
        Args:
            model_output (dict): Output of the model
            batch (torch_geometric.data.Batch): Batch of data
        Returns:
            torch.Tensor: Objective Loss
            torch.Tensor: Potential energy
            torch.Tensor: Novel loss
            torch.Tensor: Lennard-Jones repulsion
        """
        if self.normalize:
            model_output['potential_energy'] = (model_output['potential_energy'] + self.offset) / batch.n_atoms.unsqueeze(1)
            # for i in range(len(batch.n_atoms)):
            #     model_output['potential_energy'][i] = (model_output['potential_energy'][i] + self.offset[i]) / batch.n_atoms[i]
        ljr = self.lj_repulsion(batch, power=self.ljr_power)
        embedding_loss = torch.cdist(model_output['embeddings'], self.embeddings, p=2)
        if self.mode == 'min':
            embedding_loss = torch.min(embedding_loss, dim=-1, keepdim=True)[0]
            embedding_loss = torch.mean(embedding_loss, dim=0)
        else:
            embedding_loss = torch.mean(embedding_loss, dim=0)
            embedding_loss = torch.mean(embedding_loss, dim=-1, keepdim=True)
        return self.energy_ratio * model_output["potential_energy"] - self.embedding_ratio * embedding_loss + self.ljr_ratio * ljr, model_output["potential_energy"], -embedding_loss, ljr