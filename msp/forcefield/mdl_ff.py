from msp.forcefield.base import ForceField

import torch
from torch_geometric.data import Data
import numpy as np
import yaml
import os
from torch import distributed as dist
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
        train_ratio = self.train_config["dataset"]["train_ratio"]
        val_ratio = self.train_config["dataset"]["val_ratio"]
        test_ratio = self.train_config["dataset"]["test_ratio"]
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    dataset,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
        self.trainer = self.from_config_train(self.train_config, self.dataset)
    
        
                    
    def train(self, dataset):
        """
        Train the force field model on the dataset.
        """
        dataset = self.process_data(dataset)
        dataset = dataset['full']
        train_ratio = self.train_config["dataset"]["train_ratio"]
        val_ratio = self.train_config["dataset"]["val_ratio"]
        test_ratio = self.train_config["dataset"]["test_ratio"]
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = dataset_split(
                    dataset,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                )
        self.update_trainer(self.train_config, self.dataset)
        #initialize new model
        self.model = self.trainer.model
        self.trainer.train()
        state = {"state_dict": self.model.state_dict()}
        torch.save(state, 'model/best_checkpoint.pt')

    def update(self, dataset):
        """
        Update the force field model on the dataset.
        """
        dataset = self.process_data(dataset)
        self.dataset['train'] = dataset['full']
        self.update_trainer(self.train_config, self.dataset)
        self.model = self.trainer.model
        self.trainer.train()
        state = {"state_dict": self.model.state_dict()}
        torch.save(state, 'model/best_checkpoint.pt')
    
    def process_data(self, dataset):
        """
        Process data for the force field model.
        """
        new_data_list = [Data() for _ in range(len(dataset))]
        for i, struc in enumerate(dataset):
            data = new_data_list[i]
            data.n_atoms = len(struc['atomic_numbers'])
            data.pos = torch.tensor(struc['positions'])
            data.cell = torch.tensor([struc['cell']])
            data.structure_id = [struc['structure_id']]
            data.z = torch.tensor(struc['atomic_numbers'])
            data.u = torch.tensor(np.zeros((3))[np.newaxis, ...]).float()
            if 'y' not in struc:
                if 'relaxed_energy' in struc:
                    data.y = np.array([struc['relaxed_energy']])
                elif 'energy' in struc:
                    data.y = np.array([struc['energy']])
                elif 'potential_energy' in struc:
                    data.y = np.array([struc['potential_energy']])
            else:
                data.y = np.array([struc['y']])
            data.y = torch.tensor(data.y).float()
            if data.y.dim() == 1:
                data.y = data.y.unsqueeze(0) 
        dataset = {"full": new_data_list}
        return dataset


    def forward(self, data):
        """
        Calls model directly
        """
        output = self.model(data)
        #output is a dict
        return output
        
    def create_ase_calc(self, calculator_config=None):
        """
        Returns ase calculator
        """
        calculator = MDLCalculator(config=calculator_config)        
        return calculator
    
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
    
    def update_trainer(self, config, dataset):
        """Class method used to initialize PropertyTrainer from a config object
        config has the following sections:
            trainer
            task
            model
            optim
            scheduler
            dataset
        """

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
        self.trainer.epoch = 0
        self.trainer.best_metric = 1e10
        self.trainer.optimizer = BaseTrainer._load_optimizer(config["optim"], self.trainer.model, local_world_size)
        self.trainer.train_sampler = BaseTrainer._load_sampler(config["optim"], self.dataset, local_world_size, rank)
        self.trainer.data_loader = BaseTrainer._load_dataloader(
            config["optim"],
            config["dataset"],
            dataset,
            self.trainer.train_sampler,
            config["task"]["run_mode"],
        )
        self.trainer.scheduler = BaseTrainer._load_scheduler(config["optim"]["scheduler"], self.trainer.optimizer)
        self.trainer.loss = BaseTrainer._load_loss(config["optim"]["loss"])
        
    