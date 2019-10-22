"""Ignore since removing."""
import numpy as np
import matplotlib.pyplot as plt
from flow.core.macroscopic import ARZ
from flow.core.macroscopic.arz import PARAMS as ARZ_PARAMS
import json
params = ARZ_PARAMS.copy()

json_dir = "/Users/gilbertbahatij/flow/flow/core/macroscopic/json_files/"
json_params = json.load(open(json_dir + 'arz_initial_data.json', 'r'))
params['initial_conditions'] = np.array(json_params['initial_conditions'])

# params['boundary_conditions'] = np.array(json_params['boundary_conditions'])
params['boundary_conditions'] = "loop"
# params['boundary_conditions'] = "extend_both"
# params['boundary_conditions'] = {"constant_both": ((0.45, 0), (0.56, 1))}

params["total_time"] = 660
if __name__ == "__main__":
    env = ARZ(params)
    env.run()

# if __name__ == '__main__':
#     for i in range(50):
#         action = 1
#         obs, rew, done, _ = env.step(action)

    #     # plot current profile during execution
    #     plt.plot(json_params['x'], obs[:int(obs.shape[0]/2)], 'b-')
    #     plt.axis([0, params['length'], 0.4, 0.8])
    #     plt.ylabel('Density')
    #     plt.xlabel('Street Length')
    #     plt.title("ARZ Evolution of Density")
    #     plt.draw()
    #     plt.pause(0.0001)
    #     plt.clf()
    #
    # # final plot
    # plt.plot(json_params['x'], obs[:int(obs.shape[0]/2)], 'b-')
    # plt.axis([0, params['length'], 0.4, 0.8])
    # plt.ylabel('Density')
    # plt.xlabel('Street Length')
    # plt.title("ARZ Evolution of Density")
    # plt.show()