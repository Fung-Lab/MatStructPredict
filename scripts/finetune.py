from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
import json
import numpy as np
from msp.structure.structure_util import dict_to_atoms
import ase

if __name__ == "__main__":

    # load original dataset model was trained on
    eval_dataset = json.load(open("../data/data_subset_msp.json", "r"))

    # load dataset to finetune on
    finetune_dataset = json.load(open("../data/iter_one_structures/data.json", "r"))

    # convert stress to correct units
    for data in finetune_dataset:
        data["stress"] = np.array(data["stress"])*0.006242*-0.1

    # Select config file for forcefield
    train_config = 'mdl_config.yml'
    forcefield = MDL_FF(train_config, finetune_dataset)

    print("Evaluating before finetuning on eval_dataset")
    forcefield.validate(eval_dataset, val_ratio=1, batch_size=12)

    print("Evaluating before finetuning on finetuning_dataset")
    forcefield.validate(finetune_dataset, val_ratio=1, batch_size=12)

    # Finetune the model
    forcefield.update(finetune_dataset, .95, .05, 0, max_epochs=100, save_model=False, batch_size=12, save_path='fine_tuned_models')

    print("Evaluating after finetuning on finetuning_dataset")
    forcefield.validate(finetune_dataset, val_ratio=1, batch_size=12)

    print("Evaluating after finetuning on eval_dataset")
    forcefield.validate(eval_dataset, val_ratio=1, batch_size=12)
