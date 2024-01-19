from msp.dataset import download_dataset, load_dataset, combine_dataset, update_dataset
from msp.composition import generate_random_compositions, sample_random_composition
from msp.forcefield import MDL_FF
from msp.structure.globalopt.basin_hopping import BasinHoppingASE, BasinHopping
from msp.utils.objectives import UpperConfidenceBound, Energy
from msp.validate import read_dft_config, setup_DFT, Validate
import pickle as pkl
import json
import numpy as np
import ase
from ase import io


#download dataset from Materials Project
#return dataset class or dict
my_dataset = download_dataset(repo="MP", save=True)
#or load dataset from disk:

#my_dataset = load_dataset(path ="path/to/dataset")
my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
#print(my_dataset[0])
max_iterations=10

#Initialize a forcefield class, reading in from config (we use MDL_FF but it can be a force field from another library)
train_config = 'mdl_config.yml'
forcefield = MDL_FF(train_config, my_dataset)
#train the forcefield (optional)
forcefield.train(my_dataset, .09, .05, .05, max_epochs=1)
#to load saved model, use update and put the path to file in checkpoint_path in the train_config
#forcefield.update(my_dataset, .09, .05, .05, max_epochs=1)

#active learning loop
for i in range(0, max_iterations):
    #sample composition using a built in random sampler that checks for repeats in the dataset
    #returns a list of compositions, could be length 1 or many
    #compositions are a dictionary of {element:amount}
    #compositions = sample_random_composition(dataset=my_dataset, n=1)
    #or manually specify the list of lists:
    compositions=[[22, 8, 8]]
    

    #forcefield itself is not an ase calculator, but can be used to return the MDLCalculator class
    forcefield_calc = forcefield.create_ase_calc()
    #initialize the predictor class, this is the BasinHopping version which uses an ASE calculator, but we can have another version for batched search

    #predictor = BasinHoppingASE(forcefield_calc, hops=1, steps=100, optimizer="FIRE", dr=0.5)
    predictor = BasinHopping(forcefield, hops=3, steps=20, dr=.5)
    #alternatively if we dont use ASE, we can optimize in batch, and optimize over multiple objectives as well
    #we do this by first initializing our objective function, which is similar to the loss function class in matdeeplearn
    #objective_func = UpperConfidenceBound(c=0.1)
    objective_func = Energy()
    #---Optimizing a batch of structures with batch basin hopping---
    minima_list = predictor.predict(compositions, objective_func, batch_size=32)
    for j, minima in enumerate(minima_list):
        filename = "iteration_"+str(i)+"_structure_"+str(j)+".cif"
        ase.io.write(filename, minima)
    
    #predict structures using BasinHoppingASE
    #minima_list=[]
    #for j in range(0, len(compositions)):
    #    putative_minima = predictor.predict(compositions[j], topk=1)
    #    minima_list.append(putative_minima[0])
    
    
    #validate with DFT on-demand on the putative minima
    dft_path = 'path/to/dft_config.yml'
    dft_config=read_dft_config(dft_path)
    method = setup_DFT(dft_config)
    validator = Validate(method=method, local=False)
    dft_results=[]
    for j in range(0, len(minima_list)):
        dft_results.append(validator(minima_list[j]))
    
    
    #my_dataset = combine_dataset(my_dataset, dft_results)
    
    #retrain the forcefield
    #forcefield.train(my_dataset)
    #or finetune the forcefield rather than from scratch
    #forcefield.update(dft_results)
    forcefield.update(my_dataset, .009, .05, .05, max_epochs=1)
    
    
    #update the dataset as well
    update_dataset(repo="MP", data=dft_results)

print("Job done")
