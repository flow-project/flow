"""Ignore since removing."""
import numpy as np
from flow.core.macroscopic import LWR
from flow.core.macroscopic.lwr import PARAMS as LWR_PARAMS
import json

params = LWR_PARAMS.copy()

json_dir = "/Users/gilbertbahatij/flow/flow/core/macroscopic/json_files/"
json_params = json.load(open(json_dir + 'lwr_initial_data.json', 'r'))

params['initial_conditions'] = np.array(json_params['initial_conditions'])
# params['boundary_conditions'] = {"constant_both": (4, 2)}
params["total_time"] = 5525
params['boundary_conditions'] = "loop"

if __name__ == "__main__":
    env = LWR(params)
    env.run()
