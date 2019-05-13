
from flow.multiagent_envs.grid.grid_trafficlight_timing import MultiAgentGrid

from RL_brain import DeepQNetwork

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.scenarios.grid import SimpleGridScenario
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.controllers.routing_controllers import GridRouter

import csv
import datetime

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
        veh_id="human",
        routing_controller=(GridRouter, {}),
        num_vehicles=tot_cars)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    tl_logic = TrafficLightParams(baseline=False)
    phases = [{
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "GGGrrrGGGrrr"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "yyyrrryyyrrr"
    }, {
        "duration": "31",
        "minDur": "8",
        "maxDur": "45",
        "state": "rrrGGGrrrGGG"
    }, {
        "duration": "6",
        "minDur": "3",
        "maxDur": "6",
        "state": "rrryyyrrryyy"
    }]
    tl_logic.add("center0", phases=phases, programID=1)
    tl_logic.add("center1", phases=phases, programID=1)
    tl_logic.add("center2", tls_type="actuated", phases=phases, programID=1)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing='custom')

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

    # end of game
    print('game over')

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