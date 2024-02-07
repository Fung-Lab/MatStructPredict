from msp.dataset import download_dataset, load_dataset, combine_dataset, update_dataset
from msp.composition import generate_random_compositions, sample_random_composition
from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
from msp.structure.globalopt.basin_hopping import BasinHoppingASE, BasinHopping
from msp.utils.objectives import UpperConfidenceBound, Energy
from msp.utils import atoms_from_dict
from msp.validate import read_dft_config, setup_DFT, Validate
import pickle as pkl
import json
import numpy as np
import ase
import torch
from ase import io
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

#download dataset from Materials Project
#return dataset class or dict
my_dataset = download_dataset(repo="MP", save=True)
#or load dataset from disk:

#my_dataset = load_dataset(path ="path/to/dataset")
my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
#print(my_dataset[0])
max_iterations=1

#Initialize a forcefield class, reading in from config (we use MDL_FF but it can be a force field from another library)
train_config = 'mdl_config.yml'
forcefield = MDL_FF(train_config, my_dataset)

predictor = BasinHoppingASE(forcefield, hops=5, steps=100, optimizer="FIRE", dr=0.5)

predictor_batch = BasinHopping(forcefield, hops=5, steps=100, dr=0.5, batch_size=10)

forcefield_mace = MACE_FF()
predictor_mace = BasinHoppingASE(forcefield_mace, hops=5, steps=100, optimizer="FIRE", dr=0.5)

forcefield_m3gnet = M3GNet_FF()
predictor_m3gnet = BasinHoppingASE(forcefield_m3gnet, hops=5, steps=100, optimizer="FIRE", dr=0.5)
#train the forcefield (optional)
#forcefield.train(my_dataset, .09, .05, .05, max_epochs=1)
#to load saved model, use update and put the path to file in checkpoint_path in the train_config
#forcefield.update(my_dataset, .09, .05, .05, max_epochs=1)

#active learning loop
for i in range(0, max_iterations):
    #sample composition using a built in random sampler that checks for repeats in the dataset
    #returns a list of compositions, could be length 1 or many
    #compositions are a dictionary of {element:amount}
    #compositions = sample_random_composition(dataset=my_dataset, n=1)
    #or manually specify the list of lists:
    compositions=[[22, 22, 22, 22, 22, 22, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] for _ in range(1)]
    #compositions=[[22, 22, 8, 8, 8, 8]]
    read_structure = ase.io.read("init.cif")
    init_structures=[read_structure]

    #forcefield itself is not an ase calculator, but can be used to return the MDLCalculator class
    #initialize the predictor class, this is the BasinHopping version which uses an ASE calculator, but we can have another version for batched search
    #predictor = BasinHopping(forcefield, hops=4, steps=20, dr=.5)
    minima_list = predictor.predict(compositions, init_structures=None)
    minima_list = atoms_from_dict(minima_list)
    for j, minima in enumerate(minima_list):
        filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mdl.cif"
        ase.io.write(filename, minima)

    #---Optimizing a batch of structures with batch basin hopping---
    #alternatively if we dont use ASE, we can optimize in batch, and optimize over multiple objectives as well
    #we do this by first initializing our objective function, which is similar to the loss function class in matdeeplearn
    #objective_func = UpperConfidenceBound(c=0.1)
    objective_func = Energy()
    minima_list_batch = predictor_batch.predict(compositions, objective_func, init_structures=None, optim='Adam', batch_size=32)
    minima_list_batch = atoms_from_dict(minima_list_batch)
    for j, minima in enumerate(minima_list_batch):
        filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
        ase.io.write(filename, minima)
        
                  
    minima_list_mace = predictor_mace.predict(compositions, init_structures=None)    
    minima_list_mace = atoms_from_dict(minima_list_mace)
    for j, minima in enumerate(minima_list_mace):
        filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mace.cif"
        ase.io.write(filename, minima)
       
    
    minima_list_m3gnet = predictor_m3gnet.predict(compositions, init_structures=None)   
    minima_list_m3gnet = atoms_from_dict(minima_list_m3gnet)
    for j, minima in enumerate(minima_list_m3gnet):
        filename = "iteration_"+str(i)+"_structure_"+str(j)+"_m3gnet.cif"
        ase.io.write(filename, minima)
    
    #check if the true structure has been found (either yes or no)   
    adaptor = AseAtomsAdaptor
    structure_matcher = StructureMatcher(ltol = 0.3, stol = 0.3, angle_tol = 5, primitive_cell = True, scale = True)
    print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list[0])))
    print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_batch[0])))
    print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_mace[0])))
    print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_m3gnet[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_batch[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_mace[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_m3gnet[0]))) 
       
    #quantify structure similairy, continous from 1 to 0
    #see: https://docs.materialsproject.org/methodology/materials-methodology/related-materials
    #matminer may need older version of numpy==1.23.5
    ssf = SiteStatsFingerprint(CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0), stats=('mean', 'std_dev', 'minimum', 'maximum'))
    target = np.array(ssf.featurize(adaptor.get_structure(read_structure)))
    mdl = np.array(ssf.featurize(adaptor.get_structure(minima_list[0])))
    mdl_batch = np.array(ssf.featurize(adaptor.get_structure(minima_list_batch[0])))
    mace = np.array(ssf.featurize(adaptor.get_structure(minima_list_mace[0])))
    m3gnet = np.array(ssf.featurize(adaptor.get_structure(minima_list_m3gnet[0])))
    print('Distance between target and mdl: {:.4f}'.format(np.linalg.norm(target - mdl)))
    print('Distance between target and mdl_batch: {:.4f}'.format(np.linalg.norm(target - mdl_batch)))
    print('Distance between target and mace: {:.4f}'.format(np.linalg.norm(target - mace)))
    print('Distance between target and m3gnet: {:.4f}'.format(np.linalg.norm(target - m3gnet)))
    
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
    #forcefield.update(my_dataset, .009, .05, .05, max_epochs=1)
    
    
    #update the dataset as well
    update_dataset(repo="MP", data=dft_results)

print("Job done")
