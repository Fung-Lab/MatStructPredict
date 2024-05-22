import sys
from msp.dataset import download_dataset, load_dataset, combine_dataset, update_dataset
from msp.composition import generate_random_compositions, sample_random_composition, generate_random_lithium_compositions
from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
from msp.optimizer.globalopt.basin_hopping import BasinHoppingASE, BasinHoppingBatch
from msp.utils.objectives import EnergyAndUncertainty, Energy, EmbeddingDistance
from msp.structure.structure_util import dict_to_atoms, init_structure, atoms_to_dict
from msp.validate import read_dft_config, setup_DFT, Validate
import pickle as pkl
import json
import numpy as np
import ase
import torch
from ase import io
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import time

import matplotlib.pyplot  as plt

#download dataset from Materials Project
#return dataset class or dict
my_dataset = download_dataset(repo="MP", save=True)
#or load dataset from disk:

#my_dataset = load_dataset(path ="path/to/dataset")
my_dataset = json.load(open("/global/cfs/projectdirs/m3641/Shared/Materials_datasets/MP_data_latest/raw/data.json", "r"))
predicted_structures = []
# my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
#print(my_dataset[0])
max_iterations=1

#Initialize a forcefield class, reading in from config (we use MDL_FF but it can be a force field from another library)
train_config = 'mdl_config.yml'
forcefield = MDL_FF(train_config, my_dataset)
embeddings = forcefield.get_embeddings(my_dataset, batch_size=40, cluster=False)

#predictor = BasinHoppingASE(forcefield, hops=5, steps=100, optimizer="FIRE", dr=0.5)

predictor_batch = BasinHoppingBatch(forcefield, hops=10, steps=100, dr=0.6, optimizer='Adam', perturbs=['pos', 'cell'])


# forcefield_mace = MACE_FF()
# predictor_mace = BasinHoppingASE(forcefield_mace, hops=5, steps=100, optimizer="FIRE", dr=0.5)

# forcefield_m3gnet = M3GNet_FF()
# predictor_m3gnet = BasinHoppingASE(forcefield_m3gnet, hops=5, steps=100, optimizer="FIRE", dr=0.5)
#train the forcefield (optional)
#forcefield.train(my_dataset, .09, .05, .05, max_epochs=1)
#to load saved model, use update and put the path to file in checkpoint_path in the train_config
#forcefield.update(my_dataset, .09, .05, .05, max_epochs=1)

#active learning loop
for i in range(0, max_iterations):
    # sample composition using a built in random sampler that checks for repeats in the dataset
    # returns a list of compositions, could be length 1 or many
    # compositions are a dictionary of {element:amount}
    # compositions = sample_random_composition(dataset=my_dataset, n=1)
    # or manually specify the list of lists:
    # compositions = [[22, 22, 22, 22, 22, 22, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] for _ in range(8)]

    if i != 0:
        forcefield.update(predicted_structures, 1, 0, 0, max_epochs=30, save_model=False)

    # compositions = generate_random_compositions(my_dataset, n=8, max_elements=5, max_atoms=20)
    compositions_novelty = generate_random_lithium_compositions(my_dataset, n=16000)
    initial_structures_novelty = [init_structure(c, pyxtal=True) for c in compositions_novelty]
    # compositions_energy = generate_random_lithium_compositions(my_dataset, n=4000)
    # initial_structures_energy = [init_structure(c, pyxtal=True) for c in compositions_energy]
    # for j, minima in enumerate(dict_to_atoms(initial_structures)):
    #     filename = "initial_iteration_"+str(i)+"_structure_"+str(j)+".cif"
    #     ase.io.write(filename, minima)
    # read_structure = ase.io.read("init.cif")

    # initial_structures=[atoms_to_dict([read_structure], loss=[None])]

    #forcefield itself is not an ase calculator, but can be used to return the MDLCalculator class
    #initialize the predictor class, this is the BasinHopping version which uses an ASE calculator, but we can have another version for batched search

    # total_list, minima_list = predictor.predict(initial_structures)
    # minima_list = dict_to_atoms(minima_list)
    # for j, minima in enumerate(minima_list):
    #     filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mdl.cif"
    #     ase.io.write(filename, minima)
    # f = open('output.txt', 'w')
    # for i in range(len(total_list)):
    #     f.write('Structure ' + str(i) + '\n')
    #     for hop in total_list[i]:
    #         f.write("\tHop: " +str(hop['hop'])+ '\n')
    #         f.write("\t\tInit loss: " +str(hop['init_loss'])+ '\n')
    #         f.write("\t\tFinal loss: " +str(hop['loss'])+ '\n')
    #         f.write("\t\tComposition: " +str(hop['composition'])+ '\n')
    #         f.write("\t\tperturb: " +str(hop['perturb'])+ '\n')
    # f.close()

    #---Optimizing a batch of structures with batch basin hopping---
    # alternatively if we dont use ASE, we can optimize in batch, and optimize over multiple objectives as well
    # we do this by first initializing our objective function, which is similar to the loss function class in matdeeplearn
    # objective_func_energy = Energy(normalize=True, ljr_ratio=1)
    objective_func_novelty = EmbeddingDistance(embeddings, normalize=True, energy_ratio=5, ljr_ratio=1, ljr_scale=.7, embedding_ratio=.1)
    # objective_func = EnergyAndUncertainty(normalize=True, uncertainty_ratio=.25, ljr_ratio=1, ljr_scale=.7)
    # start_time = time.time()
    # total_list_batch, minima_list_batch, best_hop, energies, accepts, accept_rate, temps, step_sizes = predictor_batch.predict(initial_structures_energy, objective_func_energy, batch_size=8, log_per=0, lr=.05)
    # top_energy = sorted(minima_list_batch, key=lambda struc: struc['objective_loss'])[:100]
    # print('---------TOP 100 ENERGY STRUCTURES---------')
    # print(top_energy)
    # print('---------TOP 100 ENERGY STRUCTURES---------')
    # minima_list_batch_ase = dict_to_atoms(minima_list_batch)
    # top_energy_ase = dict_to_atoms(top_energy)
    # for j, minima in enumerate(minima_list_batch_ase):
    #     filename = "all_4k_energy/iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
    #     ase.io.write(filename, minima)
    # for j, minima in enumerate(top_energy_ase):
    #     filename = "top_100_energy/iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
    #     ase.io.write(filename, minima)
    # f = open('output.txt', 'w')
    # for i in range(len(total_list_batch)):
    #     f.write('Structure ' + str(i) + '\n')
    #     f.write('\tbest_hop: ' + str(best_hop[j]) + '\n')
    #     for hop in total_list_batch[i]:
    #         f.write("\tHop: " +str(hop['hop'])+ '\n')
    #         f.write("\t\tObjective loss: " +str(hop['objective_loss'])+ '\n')
    #         f.write("\t\tEnergy loss: "+str(hop['energy_loss'])+'\n')
    #         if getattr(objective_func_energy, 'normalize', False):
    #             f.write("\t\tUnnormalized energy loss: " +str(hop['unnormalized_loss'])+ '\n')
    #         f.write("\t\tNovel loss: "+str(hop['novel_loss']) + '\n')
    #         f.write("\t\tSoft sphere loss: "+ str(hop['soft_sphere_loss']) + '\n')
    #         f.write("\t\tComposition: " +str(hop['composition'])+ '\n')
    #         f.write("\t\tperturb: " +str(hop['perturb'])+ '\n')
    # f.close()
    # print('Time taken for energy: {:.2f}'.format(time.time() - start_time))

    start_time = time.time()
    total_list_batch, minima_list_batch, best_hop, energies, accepts, accept_rate, temps, step_sizes = predictor_batch.predict(initial_structures_novelty, objective_func_novelty, batch_size=8, log_per=0, lr=.05)
    top_novelty = sorted(minima_list_batch, key=lambda struc: struc['objective_loss'])[:400]
    print('---------TOP 400 NOVELTY STRUCTURES---------')
    print(top_novelty)
    print('---------TOP 400 NOVELTY STRUCTURES---------')
    minima_list_batch_ase = dict_to_atoms(minima_list_batch)
    top_novelty_ase = dict_to_atoms(top_novelty)
    for j, minima in enumerate(minima_list_batch_ase):
        filename = "all_16k_novelty_5/iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
        ase.io.write(filename, minima)
    for j, minima in enumerate(top_novelty_ase):
        filename = "top_400_novelty_5/iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
        ase.io.write(filename, minima)
    # f = open('output.txt', 'w')
    # for i in range(len(total_list_batch)):
    #     f.write('Structure ' + str(i) + '\n')
    #     f.write('\tbest_hop: ' + str(best_hop[j]) + '\n')
    #     for hop in total_list_batch[i]:
    #         f.write("\tHop: " +str(hop['hop'])+ '\n')
    #         f.write("\t\tObjective loss: " +str(hop['objective_loss'])+ '\n')
    #         f.write("\t\tEnergy loss: "+str(hop['energy_loss'])+'\n')
    #         if getattr(objective_func_novelty, 'normalize', False):
    #             f.write("\t\tUnnormalized energy loss: " +str(hop['unnormalized_loss'])+ '\n')
    #         f.write("\t\tNovel loss: "+str(hop['novel_loss']) + '\n')
    #         f.write("\t\tSoft sphere loss: "+ str(hop['soft_sphere_loss']) + '\n')
    #         f.write("\t\tComposition: " +str(hop['composition'])+ '\n')
    #         f.write("\t\tperturb: " +str(hop['perturb'])+ '\n')
    # f.close()
    print('Time taken for novelty: {:.2f}'.format(time.time() - start_time))

    for i, energy_list in enumerate(energies):
        plt.scatter(range(len(energy_list)), energy_list, label=f'Structure {i + 1}',
                    color=['g' if a else 'r' for a in accepts[i]])
    plt.xlabel('Steps')
    plt.ylabel('Energies')
    plt.legend()
    plt.show()
    plt.close()

    for i, accept_rate_list in enumerate(accept_rate):
        plt.scatter(range(len(accept_rate_list)), accept_rate_list, label=f'Structure {i + 1}')
    plt.xlabel('Steps')
    plt.ylabel('Accept Rate')
    plt.legend()
    plt.show()
    plt.close()

    for i, temps_list in enumerate(temps):
        plt.scatter(range(len(temps_list)), temps_list, label=f'Structure {i + 1}')
    plt.xlabel('Steps')
    plt.ylabel('Temps')
    plt.legend()
    plt.show()
    plt.close()

    plt.scatter(range(len(step_sizes)), step_sizes)
    plt.xlabel('Steps')
    plt.ylabel('Step Sizes')
    plt.legend()
    plt.show()
    plt.close()

                  
    # minima_list_mace = predictor_mace.predict(initial_structures)    
    # minima_list_mace = dict_to_atoms(minima_list_mace)
    # for j, minima in enumerate(minima_list_mace):
    #     filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mace.cif"
    #     ase.io.write(filename, minima)
       
    
    # minima_list_m3gnet = predictor_m3gnet.predict(initial_structures)   
    # minima_list_m3gnet = dict_to_atoms(minima_list_m3gnet)
    # for j, minima in enumerate(minima_list_m3gnet):
    #     filename = "iteration_"+str(i)+"_structure_"+str(j)+"_m3gnet.cif"
    #     ase.io.write(filename, minima)
    
    #check if the true structure has been found (either yes or no)   
    #adaptor = AseAtomsAdaptor
    #structure_matcher = StructureMatcher(ltol = 0.3, stol = 0.3, angle_tol = 5, primitive_cell = True, scale = True)
    # print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list[0])))
    #print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_batch[0])))
    # print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_mace[0])))
    # print(structure_matcher.fit(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_m3gnet[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_batch[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_mace[0])))
    #print(structure_matcher.get_rms_dist(adaptor.get_structure(read_structure), adaptor.get_structure(minima_list_m3gnet[0]))) 
       
    #quantify structure similairy, continous from 1 to 0
    #see: https://docs.materialsproject.org/methodology/materials-methodology/related-materials
    #matminer may need older version of numpy==1.23.5
    #ssf = SiteStatsFingerprint(CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0), stats=('mean', 'std_dev', 'minimum', 'maximum'))
    #target = np.array(ssf.featurize(adaptor.get_structure(read_structure)))
    # mdl = np.array(ssf.featurize(adaptor.get_structure(minima_list[0])))
    #mdl_batch = np.array(ssf.featurize(adaptor.get_structure(minima_list_batch[0])))
    # mace = np.array(ssf.featurize(adaptor.get_structure(minima_list_mace[0])))
    # m3gnet = np.array(ssf.featurize(adaptor.get_structure(minima_list_m3gnet[0])))
    # print('Distance between target and mdl: {:.4f}'.format(np.linalg.norm(target - mdl)))
    #print('Distance between target and mdl_batch: {:.4f}'.format(np.linalg.norm(target - mdl_batch)))
    # print('Distance between target and mace: {:.4f}'.format(np.linalg.norm(target - mace)))
    # print('Distance between target and m3gnet: {:.4f}'.format(np.linalg.norm(target - m3gnet)))
    
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
    # for j in range(0, len(minima_list)):
    #     dft_results.append(validator(minima_list[j]))
    
    
    #my_dataset = combine_dataset(my_dataset, dft_results)
    
    #retrain the forcefield
    #forcefield.train(my_dataset)
    #or finetune the forcefield rather than from scratch
    #forcefield.update(dft_results)
    #forcefield.update(my_dataset, .009, .05, .05, max_epochs=1)
    
    
    #update the dataset as well
    predicted_structures.extend(minima_list_batch)

print("Job done")
