import pickle as pk
import os

import sys
print(sys.path[0])

import numpy as np
from pytest import param
from _DataGenerator import DataGenerator
from _ExperimentRunner import ExperimentRunner
from _Evaluator import Evaluator
from _Visualizer import Visualizer
import _Scalers as Scalers
import _utils as utils
from hashlib import md5


## Parameters
data_params = {
        "n_repetitions": 3,
        "graphs": ["ER-2"],
        "noise_distributions": [
            # utils.NoiseDistribution("gauss", (1, 1)),
            # utils.NoiseDistribution("gauss", (0.5, 2)),
            utils.NoiseDistribution("polynomial-2-gauss", (0.5, 2)),
            # utils.NoiseDistribution("mlp", (0.5, 2)),
            # utils.NoiseDistribution("exp", (0.5, 2)),
            # utils.NoiseDistribution("gumbel", (0.5, 2)),
        ],
        "edge_weights":[(.5, 2)],
        "n_nodes": [5, 10],
        "n_obs": [1000],
}

params = {
        'MEC': False,
        'exp_dir': os.path.join("src", "experiments"),
        'name': "test_exp_folder",
        'thres': 0.3,
        'thres_type': "standard",
        'algorithms': [
            #'golemEV_golemNV_orig',
            #'notearsLinear',
            'sortnregressIC',
            #'sortnregressIC_C',
            #'sortnregressPOLY',
        ]
    }

params['base_dir'] = os.path.join(params['exp_dir'], params['name'])

scaler = Scalers.Identity()

#create folders
utils.set_random_seed(2)
utils.create_folder(params['exp_dir'])
utils.create_folder(params['base_dir'])

### Generate data

dg = DataGenerator(params["name"], params["base_dir"], scaler=scaler)

dg.generate_and_save(**data_params)


# ### Check output

# rel_path = "experiments/test_exp_folder/test3/_data/ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000"
# fname = "ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000_1.pk"

# rel_path = "experiments/test_exp_folder/test_exp/_data/ER-2_gauss_(0.5, 2)_(0.5, 2)_5_1000"
# fname = "ER-2_gauss_(0.5, 2)_(0.5, 2)_5_1000_1.pk"

# print(os.path.join(sys.path[0], fname))

# with open("/home/silas/Documents/git/VarsortabilityExperimentSuite/src/experiments/default/default_raw/_data/ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000/ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000_0.pk", "rb") as f:
#     dataset = pk.load(f)

# print(np.max(dataset.data))


# ### Get results of algorithms on data
# max_mem = 4 * 1024 * 1024 * 1024  # 4GB
# expR = ExperimentRunner(params['name'], 
#                         params['base_dir'], 
#                         overwrite_prev=False,       
#                         num_cpus=5, 
#                         _redis_max_memory=max_mem, 
#                         use_ray=False)
# expR.sortnregressIC()
# expR.sortnregressPOLY()


# ### Evaluate results based on measures
# ev = Evaluator(params['name'], params['base_dir'], overwrite_prev=False, threshold=params['thres'], MEC=params['MEC'])
# ev.evaluate(thresholding=params['thres_type'])


# import pandas as pd 

# with open('/home/silas/Documents/git/VarsortabilityExperimentSuite/src/experiments/default/default_raw/_eval/standard_0.3.csv', "r") as file:
#     evals = pd.read_csv(file)
    
#     print(evals['W_true'].iloc[10])


# ### Visualize evalution

# n_nodes = max(data_params['n_nodes'])
# viz = Visualizer(params['name'], params['base_dir'], overwrite_prev=False, thres_type=params['thres_type'], n_nodes=n_nodes, threshold=params['thres'])

# viz.boxplot(acc_measure="shd", filters={
#             "n_nodes": n_nodes, 
#             "noise": "polynomial-2-gauss", 
#             "noise_variance": "(0.5, 2)",
#             "graph": "ER-2"})