from msp.forcefield.base import ForceField

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import yaml
import os
import copy
import gc
from torch import distributed as dist
from ase import Atoms
from matdeeplearn.common.registry import registry
from matdeeplearn.common.ase_utils import MDLCalculator
from matdeeplearn.preprocessor.processor import process_data
from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.trainers.property_trainer import PropertyTrainer
from matdeeplearn.common.data import dataset_split


class MDL_FF(ForceField):

    def __init__(self, train_config, dataset):
        """
        Initialize the surrogate model.
        """
        if isinstance(train_config, str):
            with open(train_config, "r") as yaml_file:
                self.train_config = yaml.safe_load(yaml_file)     
        #to be added
        self.dataset = {}
        dataset = self.process_data(dataset)
        dataset = dataset['full']
        self.dataset["train"] = dataset
        self.trainer = self.from_config_train(self.train_config, self.dataset)
        #self.model = self.trainer.model
    
        
                    
    def train(self, dataset, train_ratio, val_ratio, test_ratio, max_epochs=None, lr=None, batch_size=None, save_path='saved_model'):
        """
        Train the force field model on the dataset.
        """
        dataset = self.process_data(dataset)
        dataset = dataset['full']
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    dataset,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
        #temporary solution to initialize an entirely new model needed
        self.update_trainer(self.dataset, max_epochs, lr, batch_size)
        #self.model = self.trainer.model
        self.trainer.train()
        
        #state = {"state_dict": self.model.state_dict()}
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(self.trainer.model)):
            sub_path = os.path.join(save_path, f"checkpoint_{i}",)        
            os.makedirs(sub_path, exist_ok=True)        
            if str(self.trainer.rank) not in ("cpu", "cuda"):
                state = {"state_dict": self.trainer.model[i].module.state_dict()}         
            else:   
                state = {"state_dict": self.trainer.model[i].state_dict()}         
            model_path = os.path.join(sub_path, "best_checkpoint.pt")  
            torch.save(state, model_path)
        
        gc.collect()
        torch.cuda.empty_cache()
    

    def update(self, dataset, train_ratio, val_ratio, test_ratio, max_epochs=None, lr=None, batch_size=None, save_path='saved_model'):
        """
        Update the force field model on the dataset.
        """
        dataset = self.process_data(dataset)
        dataset = dataset['full']
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    dataset,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
        self.update_trainer(self.dataset, max_epochs, lr, batch_size)
        #self.model = self.trainer.model
        self.trainer.train()

  
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(self.trainer.model)):
            sub_path = os.path.join(save_path, f"checkpoint_{i}",)        
            os.makedirs(sub_path, exist_ok=True)        
            if str(self.trainer.rank) not in ("cpu", "cuda"):
                state = {"state_dict": self.trainer.model[i].module.state_dict()}         
            else:   
                state = {"state_dict": self.trainer.model[i].state_dict()}         
            model_path = os.path.join(sub_path, "best_checkpoint.pt")  
            torch.save(state, model_path)
            
        gc.collect()
        torch.cuda.empty_cache()
            
    def process_data(self, dataset):
        """
        Process data for the force field model.
        """
        #add tqdm
        new_data_list = [Data() for _ in range(len(dataset))]
        for i, struc in enumerate(dataset):
            data = new_data_list[i]
            data.n_atoms = len(struc['atomic_numbers'])
            data.pos = torch.tensor(struc['positions'])
            #check cell dimensions
            data.cell = torch.tensor([struc['cell']])
            #structure id optional or null
            if 'structure_id' in struc:
                data.structure_id = [struc['structure_id']]
            else:
                data.structure_id = [str(i)]
            data.z = torch.tensor(struc['atomic_numbers'])
            data.forces = torch.tensor(struc['forces'])
            data.stress = torch.tensor(struc['stress'])
            #optional
            data.u = torch.tensor(np.zeros((3))[np.newaxis, ...]).float()
            data.y = torch.tensor(np.array([struc['potential_energy']])).float()
            if data.y.dim() == 1:
                data.y = data.y.unsqueeze(0)
            #if forces:
            #    data.forces = torch.tensor(struc['forces'])
            #    if 'stress' in struc:
            #        data.stress = torch.tensor(struc['stress']) 
        dataset = {"full": new_data_list}
        return dataset


    def _forward(self, batch_data):
        """
        Calls model directly
        """    
        out_list = []
        for i in range(len(self.trainer.model)):
            out_list.append(self.trainer.model[i](batch_data))
                        
        out_stack = torch.stack([o["output"] for o in out_list])
        output = {}
        output["potential_energy"] = torch.mean(out_stack, dim=0)          
        output["potential_energy_uncertainty"] = torch.std(out_stack, dim=0)          
        #output is a dict        
        return output                
        
    def create_ase_calc(self):
        """
        Returns ase calculator
        """
        calculator = MDLCalculator(config=self.train_config)        
        return calculator

    def optimize(self, atoms, steps, objective_func, log_per, learning_rate, num_structures=-1, batch_size=4, device='cpu'):
        data_list = self.atoms_to_data(atoms)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = False

        # Created a data list
        loader = DataLoader(data_list, batch_size=batch_size)
        loader_iter = iter(loader)
        res_atoms = []
        res_energy = []                      
        
        for i in range(len(loader_iter)):
            batch = next(loader_iter).to(device)
            pos, cell = batch.pos, batch.cell
            
            opt = torch.optim.LBFGS([pos, cell], lr=learning_rate, max_iter=25)

            pos.requires_grad_(True)
            cell.requires_grad_(True)

            def closure(step, temp):
                opt.zero_grad()
                output = self._forward(batch.to(device)) 
                loss = objective_func(output)
                loss.mean().backward(retain_graph=True)
                
                #energies = self._forward(batch.to(device))["potential_energy"]          
                #mean_energy = energies.mean()
                #mean_energy.backward(retain_graph=True)
                
                if log_per > 0 and step[0] % log_per == 0:
                    print("{0:4d}   {1: 3.6f}".format(step[0], loss.mean().item()))
                step[0] += 1
                batch.pos, batch.cell = pos, cell
                temp[0] = loss
                return loss.mean()

            temp = [0]
            step = [0]
            for _ in range(steps):
                opt.step(lambda: closure(step, temp))
            res_atoms.extend(self.data_to_atoms(batch))
            res_energy.extend(temp[0].cpu().detach().numpy())

                
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = True           
             
        return res_atoms, res_energy
    
    def atoms_to_data(self, atoms):
        n_structures = len(atoms)
        data_list = [Data() for _ in range(n_structures)]

        for i, s in enumerate(atoms):
            data = atoms[i]

            pos = torch.tensor(data.get_positions(), dtype=torch.float)
            cell = torch.tensor(np.array([data.cell[:]]), dtype=torch.float)
            atomic_numbers = torch.LongTensor(data.numbers)
            structure_id = 0
                    
            data_list[i].n_atoms = len(atomic_numbers)
            data_list[i].pos = pos
            data_list[i].cell = cell   
            data_list[i].structure_id = [structure_id]  
            data_list[i].z = atomic_numbers
            data_list[i].u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
        return data_list

    def data_to_atoms(self, batch):
        res = []
        curr = 0
        for i in range(len(batch.n_atoms)):
            res.append(Atoms(batch.z[curr:curr+batch.n_atoms[i]].cpu().numpy(), cell=batch.cell[i].cpu().detach().numpy(), pbc=(True, True, True), positions=batch.pos[curr:curr+batch.n_atoms[i]].cpu().detach().numpy()))
            curr += batch.n_atoms[i]
        return res
    
    def from_config_train(self, config, dataset):
        """Class method used to initialize PropertyTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
            scheduler
            dataset
        """
        #BaseTrainer.set_seed(config["task"].get("seed"))

        if config["task"]["parallel"] == True:
            local_world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
            local_world_size = int(local_world_size)
            dist.init_process_group(
                "nccl", world_size=local_world_size, init_method="env://"
            )
            rank = int(dist.get_rank())
        else:
            rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_world_size = 1
        model = BaseTrainer._load_model(config["model"], config["dataset"]["preprocess_params"], self.dataset, local_world_size, rank)
        optimizer = BaseTrainer._load_optimizer(config["optim"], model, local_world_size)
        sampler = BaseTrainer._load_sampler(config["optim"], self.dataset, local_world_size, rank)
        data_loader = BaseTrainer._load_dataloader(
            config["optim"],
            config["dataset"],
            self.dataset,
            sampler,
            config["task"]["run_mode"],
            config["model"],
        )     
        
        scheduler = BaseTrainer._load_scheduler(config["optim"]["scheduler"], optimizer)
        loss = BaseTrainer._load_loss(config["optim"]["loss"])
        max_epochs = config["optim"]["max_epochs"]
        clip_grad_norm = config["optim"].get("clip_grad_norm", None)
        verbosity = config["optim"].get("verbosity", None)
        batch_tqdm = config["optim"].get("batch_tqdm", False)
        write_output = config["task"].get("write_output", [])
        output_frequency = config["task"].get("output_frequency", 0)
        model_save_frequency = config["task"].get("model_save_frequency", 0)
        max_checkpoint_epochs = config["optim"].get("max_checkpoint_epochs", None)
        identifier = config["task"].get("identifier", None)

        # pass in custom results home dir and load in prev checkpoint dir
        save_dir = config["task"].get("save_dir", None)
        checkpoint_path = config["task"].get("checkpoint_path", None)

        if local_world_size > 1:
            dist.barrier()
            
        return PropertyTrainer(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            sampler=sampler,
            scheduler=scheduler,
            data_loader=data_loader,
            loss=loss,
            max_epochs=max_epochs,
            clip_grad_norm=clip_grad_norm,
            max_checkpoint_epochs=max_checkpoint_epochs,
            identifier=identifier,
            verbosity=verbosity,
            batch_tqdm=batch_tqdm,
            write_output=write_output,
            output_frequency=output_frequency,
            model_save_frequency=model_save_frequency,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path,
            use_amp=config["task"].get("use_amp", False),
        )
    
    def update_trainer(self, dataset, max_epochs=None, lr=None, batch_size=None):
        """Class method used to initialize PropertyTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
            scheduler
            dataset
        """

        if self.train_config["task"]["parallel"] == True:
            local_world_size = os.environ.get("LOCAL_WORLD_SIZE", None)
            local_world_size = int(local_world_size)
            dist.init_process_group(
                "nccl", world_size=local_world_size, init_method="env://"
            )
            rank = int(dist.get_rank())
        else:
            rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_world_size = 1
        self.trainer.epoch = 0
        #self.trainer.best_metric = 1e10
        if lr is not None:
            self.train_config["optim"]["lr"] = lr
        if batch_size is not None:
            self.train_config["optim"]["batch_size"] = batch_size
        if max_epochs is not None:
            self.trainer.max_epochs = max_epochs
        self.trainer.optimizer = BaseTrainer._load_optimizer(self.train_config["optim"], self.trainer.model, local_world_size)
        self.trainer.train_sampler = BaseTrainer._load_sampler(self.train_config["optim"], self.dataset, local_world_size, rank)
        self.trainer.data_loader = BaseTrainer._load_dataloader(
            self.train_config["optim"],
            self.train_config["dataset"],
            dataset,
            self.trainer.train_sampler,
            self.train_config["task"]["run_mode"],
            self.train_config["model"],
        )
        self.trainer.scheduler = BaseTrainer._load_scheduler(self.train_config["optim"]["scheduler"], self.trainer.optimizer)
        self.trainer.loss = BaseTrainer._load_loss(self.train_config["optim"]["loss"])
        
    def load_saved_model(self, save_path):
        """Loads the model from a checkpoint.pt file"""

        # Load params from checkpoint
        for i in range(len(self.trainer.model)):
            model_path = os.path.join(save_path, f"checkpoint_{i}", "best_checkpoint.pt")    
            checkpoint = torch.load(model_path)     
            if str(self.trainer.rank) not in ("cpu", "cuda"):
                self.trainer.model[i].module.load_state_dict(checkpoint["state_dict"])
                self.trainer.best_model_state[i] = copy.deepcopy(self.trainer.model[i].module.state_dict())
            else:
                self.trainer.model[i].load_state_dict(checkpoint["state_dict"])
                self.trainer.best_model_state[i] = copy.deepcopy(self.trainer.model[i].state_dict())
        
        print("model loaded successfully")


       