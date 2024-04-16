import sys
sys.path.append(".")
#sys.path.append("/net/csefiles/coc-fung-cluster/Arpan/MatStructPredict/")

print(sys.path)
from msp.dataset import download_dataset, load_dataset, combine_dataset, update_dataset
from msp.composition import sample_random_composition
from msp.optimizer.globalopt.basin_hopping import BasinHoppingSurface
from msp.utils.objectives import UpperConfidenceBound
from msp.validate import read_dft_config, setup_DFT, Validate
from msp.forcefield.mace_ff import MACE_FF
from ase.io import read
import ase.io

forcefield = MACE_FF()

forcefield_calc = forcefield.create_ase_calc()

atoms = read("data/SrTiO3_100_surface.cif")
max_iterations=1

print(atoms)

#atoms.set_calculator(forcefield_calc)
#print(atoms.get_calculator())
#print(atoms.get_potential_energy())


for i in range(0, max_iterations): # for now just do 1 iteration
    predictor = BasinHoppingSurface(calculator=forcefield_calc, hops=5, steps=25, dr=.5)

    #compositions = sample_random_composition(dataset=atoms, n=1)
    compositions=[atoms.get_chemical_formula()]
    print("compositions: ", compositions)
    #compositions=[{'Ti':2, 'O':1}, {'Al':2, 'O':3}]
    #print(compositions)
    minima_list = predictor.predict(atoms)
    print("minima list is ", minima_list)
    print(type(minima_list))
    print(minima_list.get_chemical_formula())
    # write minima list to cif file to see structure

    dft_path = 'path/to/dft_config.yml'
    dft_config = read_dft_config(dft_path)
    method = setup_DFT(dft_config)
    validator = Validate(method=method, local=False)
    dft_results = []
    for atom in minima_list:
        dft_results.append(validator(atom))

    # need to make it similar to example.py
    filename = "iteration_"+str(i)+"_structure_"+str(0)+"_mdl.cif"
    ase.io.write(filename, minima_list)
   # for j, minima in enumerate(minima_list):
   #     filename = "iteration_"+str(i)+"_structure_"+str(j)+"_mdl.cif"
   #     ase.io.write(filename, minima)

# Update the dataset as well
print("dft results: ", dft_results)
update_dataset(repo="MP", data=dft_results)

print("Job done")
