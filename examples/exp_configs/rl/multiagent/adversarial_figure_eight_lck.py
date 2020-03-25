"""Example of a multi-agent environment containing a figure eight.

This example consists of one autonomous vehicle and an adversary that is
allowed to perturb the accelerations of figure eight.
"""

# WARNING: Expected total reward is zero as adversary reward is
# the negative of the AV reward

from copy import deepcopy
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import ContinuousRouter
from flow.controllers import IDMController  		 #the human vehicles follows a physics algo
from flow.controllers import RLController  	 		 #the av drives according to RL algo
from flow.controllers import SimLaneChangeController #Controller to enforce sumo lane-change dynamics 
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams,SumoLaneChangeParams #lane change params
from flow.core.params import VehicleParams
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS #specifies the env structure (for example, the number of lanes)
from flow.envs.multiagent import AdversarialAccelEnv  #This is new!
from flow.networks import FigureEightNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# time horizon of a single rollout
HORIZON    = 1000 #1500  #how many steps on each game
# number of rollouts per training iteration
N_ROLLOUTS = 2# 4  #how many steps on a learning batch
# number of parallel workers
N_CPUS     = 44
# number of human-driven vehicles
N_HUMANS   = 22 #increased number of human cars to make it more dramatic
# number of automated vehicles
#N_AVS      = 2 -> will be defined later as I need different colors for each car

# ContinuousRouter controller -> to perpetually maintain the vehicle within the network.
# lane_change_controller=(SimLaneChangeController, {}) -> used to enforce sumo lane-change dynamics on a vehicle.
# lane_change_params=SumoLaneChangeParams(lane_change_mode="speed gain",) - > cars can change lane

vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode='strategic',),
    acceleration_controller=(IDMController, {'noise': 0.2}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(speed_mode='obey_safe_speed',),
    num_vehicles=N_HUMANS)

# Teh car color is overridden by flow/flow/core/kernel/vehicle/traci.py
# The reason it overrides the color is because the car's color doesnt make into the "params.json" file (it should be stored under the dict "type_params")
# When the visualizer calss "traci.py" to generate the environment, it picks the params stored on the 'params.json' file,
# since there is no color for the car on this file,  traci.py will assign car colors based on RL/human lables.
# The function doing this is:  def update_vehicle_colors(self):

#RL agent1 -
vehicles.add(
    veh_id='rl1',
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode='strategic',),
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(speed_mode='obey_safe_speed',),
    num_vehicles=1,
    color='white')

#RL agent2
vehicles.add(
    veh_id='rl2',
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode='strategic',),
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(speed_mode='obey_safe_speed',),
    num_vehicles=1,
    color='white')

flow_params = dict(
    # name of the experiment
    exp_tag='adversarial_figure_eight',

    # name of the flow environment the experiment is running on #NEW!
    env_name=AdversarialAccelEnv,  

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    #here: /home/zeta/flow/flow/core/params.py
    #force_color_update = False Do not color cars according to type
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance = True,
        force_color_update= False,
        emission_path    ='/home/zeta/Desktop/Lucia/emissions_adv_fig8',
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 20,
            'max_accel': 3,
            'max_decel': 3,
            'perturb_weight':0.9, #0.03, #weight of the adversarial agent
            'sort_vehicles': False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


create_env, env_name = make_create_env(params=flow_params, version=0) # Tutorial 3: returns a function create_env that initializes a Gym environment corresponding to the Flow network specified.

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy(), 'adversary': gen_policy()}


def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id
