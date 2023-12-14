from msp.dataset import download_dataset, load_dataset
from msp.composition import sample_composition_random
from msp.surrogate import Surrogate
from msp.structure import Prediction
from msp.structure.globalopt import BasinHopping
from msp.validate import Validate

from matdeeplearn.common.ase_utils import MDLCalculator

#download dataset from Materials Project
#return dataset class or dict
my_dataset = download_dataset(repo="MP", save=True)
#or load dataset from disk:
#my_dataset = load_dataset(path ="my_path")


#sample composition using a built in random sampler that checks for repeats in the dataset
#returns a list of compositions, could be length 1 or many
compositions = sample_composition_random(dataset=my_dataset, topk=5)
compositions=["TiO2"]

#train the surrogate (optional)
matdeeplearn.train(my_dataset)
surrogate.train(my_dataset)

#get ase calculator (can be any valid ase calculator)
calc_str = './configs/config_calculator.yml'
calculator = MDLCalculator(config=calc_str)

#Surrogate class that takes in calculator as an argument, but may have additional functions
surrogate = Surrogate()


basinhop = BasinHopping(calculator=MDLCalculator, hops=5, steps=100, optimizer="FIRE")
#predict structure returns a list of minima, could be 1 or many
predictor = Prediction(Surrogate, method = basinhop)

minima_list=[]
for i in range(0, len(compositions)):
    putative_minima = predictor.predict(compositions[i], topk=1)
    minima_list.append(putative_minima[0])

#validate with DFT on-demand
dft_config=read_dft_config()
method = setup_DFT(dft_config)
validator = Validate(method=method, local=False)
dft_results=[]
for i in range(0, len(minima_list)):
    dft_results.append(validator(minima_list[i]))

my_dataset_updated = combine_dataset(my_dataset, dft_results)

#update is a finetuning method, not from scratch
matdeeplearn.update(my_dataset_updated)
surrogate.update(my_dataset_updated)

update_dataset(repo="MP", dft_results)
