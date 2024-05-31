from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
import json


my_dataset = json.load(open("../data/iter_one_structures/data.json", "r"))

train_config = 'mdl_config.yml'
forcefield = MDL_FF(train_config, my_dataset)


forcefield.update(my_dataset, 1, 0, 0, max_epochs=30, save_model=True, batch_size=48, save_path='fine_tuned_models')