"""Ignore since removing."""
import numpy as np
from flow.core.macroscopic import ARZ
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS
import json

params = ARZ_PARAMS.copy()

# read initial data from json file
json_dir = "./json_files/"
json_params = json.load(open(json_dir + 'arz_initial_data.json', 'r'))

params['initial_conditions'] = np.array(json_params['initial_conditions'])
params["total_time"] = 66.0

params['boundary_conditions'] = "loop"
# params['boundary_conditions'] = "extend_both"
# params['boundary_conditions'] = {"constant_both": ((0.45, 0), (0.56, 1))}

if __name__ == "__main__":
    env = ARZ(params)
    env.run()
