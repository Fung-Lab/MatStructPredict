import sys
from msp.composition import generate_random_compositions, sample_random_composition, generate_random_lithium_compositions
from msp.forcefield import MDL_FF, MACE_FF, M3GNet_FF
from msp.optimizer.globalopt.basin_hopping import BasinHoppingASE, BasinHoppingBatch
from msp.utils.objectives import EnergyAndUncertainty, Energy, EmbeddingDistance
from msp.structure.structure_util import dict_to_atoms, init_structure, atoms_to_dict
import json
import ase
from ase import io
import time

import matplotlib.pyplot  as plt

if __name__ == "__main__":

    # load dataset
    my_dataset = json.load(open("/global/cfs/projectdirs/m3641/Shared/Materials_datasets/MP_data_latest/raw/data.json", "r"))
   
    # Set number of active learning iterations
    max_iterations = 1

    # Initiliaze the list of predicted structures
    predicted_structures = []

    # Initialize a forcefield class, reading in from config (we use MDL_FF but it can be a force field from another library)
    train_config = 'mdl_config.yml'
    forcefield = MDL_FF(train_config, my_dataset)

    # get embeddings for the dataset if using EmbeddingDistance
    embeddings = forcefield.get_embeddings(my_dataset, batch_size=40, cluster=False)

    # initialize the predictor class, this is the BasinHopping version which uses an ASE calculator, but we can have another version for batched optimization
    # predictor = BasinHoppingASE(forcefield, hops=5, steps=100, optimizer="FIRE", dr=0.5)
    predictor_batch = BasinHoppingBatch(forcefield, hops=50, steps=100, dr=0.6, optimizer='Adam', perturbs=['pos', 'cell'])

    # train the forcefield
    # forcefield.train(my_dataset, .09, .05, .05, max_epochs=1)

    #active learning loop
    for i in range(0, max_iterations):
        print("Iteration: ", i)

        # update the forcefield with the predicted structures
        if i != 0:
            forcefield.update(predicted_structures, 1, 0, 0, max_epochs=30, save_model=False)

        # Generate compositions for the initial structures, random or preset
        compositions = generate_random_lithium_compositions(my_dataset, n=10)

        # Generate initial structures for the compositions
        initial_structures = [init_structure(c, pyxtal=False) for c in compositions]

        # write the initial structures to file
        for j, minima in enumerate(dict_to_atoms(initial_structures)):
            filename = "initial_structures/iteration_"+str(i)+"_structure_"+str(j)+".cif"
            ase.io.write(filename, minima)


        #-----Optimizing the initial structures using BasinHopping-----

        # Set an objective function, here we use Energy
        objective_func = Energy(normalize=True, ljr_ratio=1, optimize_z=True)
       
        # Run the prediction
        start_time = time.time()
        total_list_batch, minima_list_batch, best_hop, energies, accepts, accept_rate, temps, step_sizes = predictor_batch.predict(initial_structures, objective_func, batch_size=8, log_per=0, lr=.05)
        sorted_results = sorted(minima_list_batch, key=lambda struc: struc['objective_loss'])
        # Save the structures to file
        minima_list_batch_ase = dict_to_atoms(minima_list_batch)
        for j, minima in enumerate(minima_list_batch_ase):
            filename = "predicted_structures/iteration_"+str(i)+"_structure_"+str(j)+"_mdl_batch.cif"
            ase.io.write(filename, minima)

        # Optionally save all optimization information to file
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

        print('Time taken for prediction: {:.2f}'.format(time.time() - start_time))


        # Plotting the optimization information
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
 
        predicted_structures.extend(minima_list_batch)

    print("Job done")
