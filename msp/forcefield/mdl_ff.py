from msp.forcefield.base import ForceField

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import yaml
import os
import copy
import gc
import time
from torch import distributed as dist
from torch.func import stack_module_state
from torch.func import functional_call
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ase import Atoms
from matdeeplearn.common.registry import registry
from matdeeplearn.common.ase_utils import MDLCalculator
from matdeeplearn.preprocessor.processor import process_data
from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.trainers.property_trainer import PropertyTrainer
from matdeeplearn.common.data import dataset_split
from msp.structure.structure_util import atoms_to_data, data_to_atoms


class MDL_FF(ForceField):

    def __init__(self, train_config, dataset):
        """
        Initializes the surrogate model.
        Args:
            train_config (str): Path to the training configuration file.
            dataset (dict): A dictionary of the dataset.
        Returns:
            None

        """
        if isinstance(train_config, str):
            with open(train_config, "r") as yaml_file:
                self.train_config = yaml.safe_load(yaml_file)     
        #to be added
        self.dataset = self.process_data(dataset)
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    self.dataset['full'],
                    self.train_config['dataset']['train_ratio'],
                    self.train_config['dataset']['val_ratio'],
                    self.train_config['dataset']['test_ratio'],
                )
        self.trainer = self.from_config_train(self.train_config, self.dataset)
        
    
        
                    
    def train(self, dataset, train_ratio, val_ratio, test_ratio, max_epochs=None, lr=None, batch_size=None, save_path='saved_model'):
        """
        Train the force field model on the dataset.
        Args:
            dataset (dict): A dictionary of the dataset.
            train_ratio (float): The ratio of the dataset to use for training.
            val_ratio (float): The ratio of the dataset to use for validation.
            test_ratio (float): The ratio of the dataset to use for testing.
            max_epochs (int): The maximum number of epochs to train the model. Defaults to value in the training configuration file.
            lr (float): The learning rate for the model. Defaults to value in the training configuration file.
            batch_size (int): The batch size for the model. Defaults to value in the training configuration file.
            save_path (str): The path to save the model. Defaults to 'saved_model'.
        Returns:
            None
        """
        dataset = self.process_data(dataset)
        dataset = dataset['full']
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    dataset,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
        self.trainer = self.from_config_train(self.train_config, self.dataset, max_epochs, lr, batch_size)
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
        Updates the force field model on the dataset. (Essentially finetunes model on new data)
        Args:
            dataset (dict): A dictionary of the dataset.
            train_ratio (float): The ratio of the dataset to use for training.
            val_ratio (float): The ratio of the dataset to use for validation.
            test_ratio (float): The ratio of the dataset to use for testing.
            max_epochs (int): The maximum number of epochs to train the model. Defaults to value in the training configuration file.
            lr (float): The learning rate for the model. Defaults to value in the training configuration file.
            batch_size (int): The batch size for the model. Defaults to value in the training configuration file.
            save_path (str): The path to save the model. Defaults to 'saved_model'.
        Returns:
            None
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
        Args:
            dataset (dict): A dictionary of the dataset.
        Returns:
            dict: A dictionary of the processed dataset.
        """
        #add tqdm
        new_data_list = [Data() for _ in range(len(dataset))]
        for i, struc in enumerate(dataset):
            data = new_data_list[i]
            data.n_atoms = len(struc['atomic_numbers'])
            data.pos = torch.tensor(struc['positions'])
            #check cell dimensions
            #data.cell = torch.tensor([struc['cell']])
            data.cell = torch.tensor(np.array(struc['cell']), dtype=torch.float).view(1, 3, 3)
            if (np.array(data.cell) == np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])).all():
                data.cell = torch.zeros((3,3)).unsqueeze(0)
            #structure id optional or null
            if 'structure_id' in struc:
                data.structure_id = [struc['structure_id']]
            else:
                data.structure_id = [str(i)]
            data.structure_id = [struc['structure_id']]
            data.z = torch.LongTensor(struc['atomic_numbers'])
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
        Args:
            batch_data (torch_geometric.data.Data): A batch of data.
        Returns:
            dict: A dictionary of the model output.
        """    
        out_list = []
        for i in range(len(self.trainer.model)):
            out_list.append(self.trainer.model[i](batch_data))
                        
        out_stack = torch.stack([o["output"] for o in out_list])
        embed_stack = torch.stack([o["embedding"] for o in out_list])
        output = {}
        output["potential_energy"] = torch.mean(out_stack, dim=0)
        output["potential_energy_uncertainty"] = torch.std(out_stack, dim=0)
        output['embeddings'] = embed_stack
        #output is a dict        
        return output

    def _batched_forward(self, batch_data):
        def fmodel(params, buffers, x):
            output = functional_call(self.base_model, (params, buffers), (x,))
            return output['output'], output['embedding']
        out_stack, embed_stack = torch.vmap(fmodel, in_dims=(0, 0, None))(self.params, self.buffers, batch_data)
        output = {}
        output["potential_energy"] = torch.mean(out_stack, dim=0)
        output["potential_energy_uncertainty"] = torch.std(out_stack, dim=0)
        output['embeddings'] = embed_stack
        #output is a dict
        return output
    
    def get_embeddings(self, dataset, batch_size):
        data_list = self.dataset['full']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = False
        self.params, self.buffers = stack_module_state(self.trainer.model)
        self.base_model = copy.deepcopy(self.trainer.model[0])
        self.base_model = self.base_model.to('meta')
        loader = DataLoader(data_list, batch_size=batch_size)
        loader_iter = iter(loader)
        embeddings = []
        start_time = time.time()
        temp = 0
        with torch.no_grad():
            for i in range(len(loader_iter)):
                batch = next(loader_iter).to(device)
                embeddings.append(self._batched_forward(batch)['embeddings'])
                if (i*len(batch)) > temp + 1000:
                    print('Structures', temp + 1, 'to', i*len(batch), 'took', time.time() - start_time)
                    temp = i * len(batch)
                    start_time = time.time()
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = True
        return torch.cat(embeddings, dim=1)


        
    def create_ase_calc(self):
        """
        Returns ase calculator using the model.
        Returns:
            MDLCalculator: An ase calculator.
        """
        calculator = MDLCalculator(config=self.train_config)        
        return calculator

    def optimize(self, atoms, steps, objective_func, log_per, learning_rate, num_structures=-1, batch_size=4, device='cpu', cell_relax=True, optim='Adam'):
        """
        Optimizes batches of structures using the force field model.
        Args:
            atoms (list): A list of ASE atoms objects.
            steps (int): The number of optimization steps.
            objective_func (function): The objective function to use for optimization.
            log_per (int): The number of steps between logging.
            learning_rate (float): The learning rate for the optimizer.
            num_structures (int): The number of structures to optimize. Defaults to -1.
            batch_size (int): The batch size for the optimizer. Defaults to 4.
            device (str): The device to use for optimization. Defaults to 'cpu'.
            cell_relax (bool): Whether to relax the cell. Defaults to True.
            optim (str): The optimizer to use. Defaults to 'Adam'.
        Returns:
            res_atoms (list): A list of optimized ASE atoms objects.
            res_energy (list): A list of the energies of the optimized structures.
            old_energy (list): A list of the energies of the initial structures.
        """
        data_list = atoms_to_data(atoms)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = False
        self.params, self.buffers = stack_module_state(self.trainer.model)
        self.base_model = copy.deepcopy(self.trainer.model[0])
        self.base_model = self.base_model.to('meta')
        # Created a data list
        loader = DataLoader(data_list, batch_size=batch_size)
        loader_iter = iter(loader)
        res_atoms = []
        obj_loss = []
        energy_loss = []
        novel_loss = []
        soft_sphere_loss = []
        
        print("device:", device)                       
        for i in range(len(loader_iter)):
            batch = next(loader_iter).to(device)
            if getattr(objective_func, 'normalize', False):
                objective_func.set_norm_offset(batch.z, batch.n_atoms)
            pos, cell = batch.pos, batch.cell
            # batch.z = batch.z.type(torch.float32)
            # optimized_z = batch.z

            opt = getattr(torch.optim, optim, 'Adam')([pos, cell], lr=learning_rate)
            lr_scheduler = ReduceLROnPlateau(opt, 'min', factor=0.8, patience=10)

            pos.requires_grad_(True)
            # optimized_z.requires_grad_(True)
            if cell_relax:
                cell.requires_grad_(True)

            temp_obj = [0]
            temp_energy = [0]
            temp_novel = [0]
            temp_soft_sphere = [0]
            step = [0]
            def closure(step, temp_obj, temp_energy, temp_novel, temp_soft_sphere, batch):
                opt.zero_grad()
                output = self._batched_forward(batch)
                objective_loss, energy_loss, novel_loss, soft_sphere_loss = objective_func(output, batch)
                objective_loss.mean().backward(retain_graph=True)
                
                curr_time = time.time() - start_time
                if log_per > 0 and step[0] % log_per == 0:                
                    #print("{}  {0:4d}   {1: 3.6f}".format(output, step[0], loss.mean().item()))  
                    if cell_relax:    
                        print("Structure ID: {}, Step: {}, LJR Loss: {:.6f}, Pos Gradient: {:.6f}, Cell Gradient: {:.6f}, Time: {:.6f}".format(len(batch.structure_id), 
                        step[0], objective_loss.mean().item(), pos.grad.abs().mean().item(), cell.grad.abs().mean().item(), curr_time))
                    else:
                        print("Structure ID: {}, Step: {}, LJR Loss: {:.6f}, Pos Gradient: {:.6f}, Time: {:.6f}".format(len(batch.structure_id), 
                        step[0], objective_loss.mean().item(), pos.grad.abs().mean().item(), curr_time))
                step[0] += 1
                batch.pos, batch.cell = pos, cell
                # batch.z = optimized_z
                temp_obj[0] = objective_loss
                temp_energy[0] = energy_loss
                temp_novel[0] = novel_loss
                temp_soft_sphere[0] = soft_sphere_loss
                return objective_loss.mean()
            for _ in range(steps):
                start_time = time.time()
                old_step = step[0]
                loss = opt.step(lambda: closure(step, temp_obj, temp_energy, temp_novel, temp_soft_sphere, batch))
                lr_scheduler.step(loss)
                # print('optimizer step time', time.time()-start_time)
                # print('steps taken', step[0] - old_step)
            #print("learning rate: ", opt.param_groups[0]['lr'])
            res_atoms.extend(data_to_atoms(batch))
            obj_loss.extend(temp_obj[0].cpu().detach().numpy())
            energy_loss.extend(temp_energy[0].cpu().detach().numpy())
            novel_loss.extend(temp_novel[0].cpu().detach().numpy())
            soft_sphere_loss.extend(temp_soft_sphere[0].cpu().detach().numpy())
            batch.z = batch.z.type(torch.int64)   
        for i in range(len(self.trainer.model)):
            self.trainer.model[i].gradient = True           

        return res_atoms, obj_loss, energy_loss, novel_loss, soft_sphere_loss
    
    def from_config_train(self, config, dataset, max_epochs=None, lr=None, batch_size=None):
        """
        Initializes PropertyTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
            scheduler
            dataset
        Args:
            config (dict): A dictionary of the configuration.
            dataset (dict): A dictionary of the dataset.
            max_epochs (int): The maximum number of epochs to train the model. Defaults to value in the training configuration file.
            lr (float): The learning rate for the model. Defaults to value in the training configuration file.
            batch_size (int): The batch size for the model. Defaults to value in the training configuration file.
        Returns:
            PropertyTrainer: A property trainer object.
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
        if lr is not None:
            self.train_config["optim"]["lr"] = lr
        if batch_size is not None:
            self.train_config["optim"]["batch_size"] = batch_size
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
        max_epochs = config["optim"]["max_epochs"] if max_epochs is None else max_epochs
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

        trainer = PropertyTrainer(
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
        use_checkpoint = config["task"].get("continue_job", False)
        self.trainer = trainer
        if use_checkpoint:
            print("Attempting to load checkpoint...")
            #trainer.load_checkpoint(config["task"].get("load_training_state", True))
            self.load_saved_model(checkpoint_path)
            print("loaded from", checkpoint_path)
            print("Recent checkpoint loaded successfully.")

        return trainer
    
    def update_trainer(self, dataset, max_epochs=None, lr=None, batch_size=None):
        """
        Updates the trainer with new parameters.
        Args:
            dataset (dict): A dictionary of the dataset.
            max_epochs (int): The maximum number of epochs to train the model. Defaults to value in the training configuration file.
            lr (float): The learning rate for the model. Defaults to value in the training configuration file.
            batch_size (int): The batch size for the model. Defaults to value in the training configuration file.
        Returns:
            None
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
        """
        Loads the model from a checkpoint.pt file
        Args:
            save_path (str): The path to the saved model.
        Returns:
            None
        """

        # Load params from checkpoint
        save_path = save_path.split(',')
        for i in range(len(self.trainer.model)):
            model_path = save_path[i]
            checkpoint = torch.load(model_path, map_location=torch.device(self.trainer.rank))     
            if str(self.trainer.rank) not in ("cpu", "cuda"):
                self.trainer.model[i].module.load_state_dict(checkpoint["state_dict"])
                self.trainer.best_model_state[i] = copy.deepcopy(self.trainer.model[i].module.state_dict())
            else:
                self.trainer.model[i].load_state_dict(checkpoint["state_dict"])
                self.trainer.best_model_state[i] = copy.deepcopy(self.trainer.model[i].state_dict())
        
        print("model loaded successfully")


       