from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
import json
import numpy as np
from msp.structure.structure_util import dict_to_atoms
import ase

eval_dataset = json.load(open("../data/data_subset_msp.json", "r"))
# get a random subset of eval_dataset
eval_dataset = eval_dataset

my_dataset = json.load(open("../data/iter_one_structures/data.json", "r"))
for data in my_dataset:
    data["stress"] = np.array(data["stress"])*0.006242*-0.1
#my_dataset = json.load(open("../data/data_subset_msp.json", "r"))

train_config = 'mdl_config.yml'



forcefield = MDL_FF(train_config, my_dataset)

print("Evaluating before finetuning on eval_dataset")
forcefield.validate(eval_dataset, val_ratio=1, batch_size=12)

print("Evaluating before finetuning on finetuning_dataset")
forcefield.validate(my_dataset, val_ratio=1, batch_size=12)

forcefield.train(my_dataset, .95, .05, 0, max_epochs=100, save_model=False, batch_size=12, save_path='fine_tuned_models')

print("Evaluating after finetuning on finetuning_dataset")
forcefield.validate(my_dataset, val_ratio=1, batch_size=12)

print("Evaluating after finetuning on eval_dataset")
forcefield.validate(eval_dataset, val_ratio=1, batch_size=12)
