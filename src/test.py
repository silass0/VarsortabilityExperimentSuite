import pickle as pk
import os

import sys
#print(sys.path[0])

import numpy as np
import _utils as utils
import _Scalers as Scalers
import _DataGenerator as DG
from hashlib import md5


#### Generate polynomial data

scaler = Scalers.Identity()

settings = {
        "n_repetitions": 3,
        "graphs": ["ER-2"],
        "noise_distributions": [
            # utils.NoiseDistribution("gauss", (1, 1)),
            #utils.NoiseDistribution("gauss", (0.5, 2)),
            utils.NoiseDistribution("polynomial-2-gauss", (0.5, 2)),
            #utils.NoiseDistribution("mlp", (0.5, 2)),
            # utils.NoiseDistribution("exp", (0.5, 2)),
            # utils.NoiseDistribution("gumbel", (0.5, 2)),
        ],
        "edge_weights":[(.5, 2)],
        "n_nodes": [5, 10],
        "n_obs": [1000],
}

dg = DG.DataGenerator("test3", os.path.join("src/experiments", "test_exp_folder"), scaler=scaler)

dg.generate_and_save(**settings)


#### Check output

rel_path = "experiments/test_exp_folder/test3/_data/ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000"
fname = "ER-2_polynomial-2-gauss_(0.5, 2)_(0.5, 2)_10_1000_1.pk"

# rel_path = "experiments/test_exp_folder/test_exp/_data/ER-2_gauss_(0.5, 2)_(0.5, 2)_5_1000"
# fname = "ER-2_gauss_(0.5, 2)_(0.5, 2)_5_1000_1.pk"

#print(os.path.join(sys.path[0], fname))

with open(os.path.join(sys.path[0], rel_path, fname), "rb") as f:
    dataset = pk.load(f)

print(dataset[2])