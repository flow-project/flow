
from flow.multiagent_envs.grid.grid_trafficlight_timing import MultiAgentGrid

from RL_brain import DeepQNetwork

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.scenarios.grid import SimpleGridScenario
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import InFlows
from flow.controllers.routing_controllers import GridRouter
from flow.core import rewards
from examples.rllib.green_wave import get_flow_params, get_non_flow_params

import csv
import datetime

USE_INFLOWS = True

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}

def create_grid_env(render=None):
    """
    creates an environment for the grid scenario.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    grid_env: 
        
    """
    v_enter = 10
    inner_length = 300
    long_length = 500
    short_length = 300
    N_ROWS = 2
    N_COLUMNS = 3
    num_agents = N_ROWS * N_COLUMNS
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    
    tot_cars = (num_cars_left + num_cars_right) * N_COLUMNS \
        + (num_cars_top + num_cars_bot) * N_ROWS

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": N_ROWS,
        "col_num": N_COLUMNS,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sim_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        routing_controller=(GridRouter, {}),
        num_vehicles=tot_cars)

    env_params = EnvParams(horizon=200, additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLightParams(baseline=False)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    if USE_INFLOWS:
	        initial_config, net_params = get_flow_params(
	            col_num=N_COLUMNS,
	            row_num=N_ROWS,
	            additional_net_params=additional_net_params)
    else:
	        initial_config, net_params = get_non_flow_params(
	            enter_speed=v_enter,
	            add_net_params=additional_net_params)

    scenario = SimpleGridScenario(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic
        )

    return env_params, sim_params, scenario, num_agents

def run_grid(writer, file):

    observation = env.reset()
    action = dict()

    for episode in range(10000000):
        # fresh env
        env.render()
        # add_travel_time_if_vehicle_departed()
        if episode % 100 == 0:
            for agent_id in observation.keys():
                # RL choose action based on observation
                action[agent_id] = RL[agent_id].choose_action(observation[agent_id])
            
        
            # RL take action and get next observation and reward
        observation_, reward, done, _ = env.step(action)
        action = action.fromkeys(action, 0)

        if episode % 10 == 0:
            writer.writerow(reward.values())
            file.flush()

        for agent_id in observation.keys():
            RL[agent_id].store_transition(observation[agent_id], action[agent_id], reward[agent_id], observation_[agent_id])

            if (episode > 100 and episode % 10 == 0):
                RL[agent_id].learn()

            # break while loop when end of this episode
            if done[agent_id]:
                break

        # swap observation
        observation = observation_

def open_file_to_write():
    # csv_file = "logs/multi-agent-grid" + str(datetime.datetime.now()) + ".csv"

    csv_file = "logs/multi-agent-grid" + ".csv"
    file = open(csv_file,'w')
    writer = csv.writer(file, delimiter=',', quotechar='"')
    return writer, file

if __name__ == "__main__":
    # maze game
    env_params, sim_params, scenario, num_agents = create_grid_env()

    env = MultiAgentGrid(env_params, sim_params, scenario)
    n_features = sum([x.shape[0] for x in env.observation_space.sample()])
    RL = dict()

    writer, file = open_file_to_write()
            
    for agent_id in range(1, num_agents+1):
        agent_name = 'intersection'+str(agent_id)
        RL[agent_name] = DeepQNetwork(2, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      agent = agent_id,
                      # output_graph=True
                      )
    run_grid(writer, file)
    # RL.plot_cost() 