import sys
sys.path.append(".")
from msp.dataset import download_dataset, load_dataset, combine_dataset, update_dataset
from msp.composition import sample_random_composition
from msp.optimizer.globalopt.basin_hopping import BasinHoppingCatalyst
from msp.utils.objectives import UpperConfidenceBound, Energy
from msp.validate import read_dft_config, setup_DFT, Validate
from msp.forcefield.mdl_ff import MDL_FF
from ase.io import read
import ase.io
import json
import torch
import os
import sys
from ase.constraints import FixAtoms

#file = open('larger_perturbations_console_output_new.txt', 'w')
#sys.stdout = file

my_dataset = json.load(open("data/data_subset_msp.json", "r"))
train_config = 'scripts/config.yml'
forcefield = MDL_FF(train_config, my_dataset)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load the saved model using the load_saved_model method
checkpoint_path = "data/best_checkpoint.pt"
forcefield.load_saved_model(checkpoint_path)

atoms = read("data/initial2.cif")
print(atoms, flush=True)
max_iterations = 1

#print(id(atoms), id(atoms))
#print(atoms)
predictor = BasinHoppingCatalyst(forcefield=forcefield, hops=20, steps=100, optimizer="Adam", dr=0.5, max_atom_num=1000, radius=1, elems_to_sample=[78])
objective_func = Energy(normalize=True)

for i in range(max_iterations):
    #print(id(atoms), id(atoms))
    compositions = [atoms.get_chemical_formula()]
    print("compositions:", compositions, flush=True)
    res, minima_list, best_hop = predictor.predict([atoms], objective_func, batch_size=1)
    print("minima list is", minima_list, " with length ", len(minima_list), flush=True)
    #print(type(minima_list))
    print("minima_list[0]: ", minima_list[0].get_chemical_formula(), flush=True)

    dft_path = 'path/to/dft_config.yml'
    dft_config = read_dft_config(dft_path)
    method = setup_DFT(dft_config)
    validator = Validate(method=method, local=False)
    dft_results = []
    for atom in minima_list:
        dft_results.append(validator(atom))
    filename = f"larger_perturbations_new.cif"
    ase.io.write(filename, minima_list[0])

    print("dft results:", dft_results, flush=True)
    update_dataset(repo="MP", data=dft_results)

print("Job done")