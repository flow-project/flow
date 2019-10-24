"""Ignore since removing."""
import numpy as np
from flow.core.macroscopic import LWR
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
import json

params = LWR_PARAMS.copy()

# read initial data from json file
json_dir = "./json_files/"
json_params = json.load(open(json_dir + 'lwr_initial_data.json', 'r'))

params['initial_conditions'] = np.array(json_params['initial_conditions'])
params["total_time"] = 55.25

params['boundary_conditions'] = "loop"
# params['boundary_conditions'] = "extend_both"
# params['boundary_conditions'] = {"constant_both": (4, 2)}

if __name__ == "__main__":
    env = LWR(params)
    env.run()
