from flow.controllers import RLController, ContinuousRouter
from flow.core.params import EnvParams
from flow.core.params import NetParams, InitialConfig
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.envs import EnergyOptSPDEnv
from flow.networks import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

# time horizon of a single rollout
HORIZON = 500
# number of rollouts per training iteration
N_ROLLOUTS = 100
# number of parallel workers
N_CPUS = 2


network_name = RingNetwork
name = "training_mpg"
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
initial_config = InitialConfig(spacing="uniform", perturbation=1)
vehicles = VehicleParams()
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             initial_speed=12,
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1)
sim_params = SumoParams(sim_step=0.1, render=False)

env_params = EnvParams(
    # length of one rollout
    horizon=HORIZON,
    additional_params={
        # maximum acceleration of autonomous vehicles
        "max_accel": 4,
        # maximum deceleration of autonomous vehicles
        "max_decel": -4,
        # bounds on the ranges of ring road lengths the autonomous vehicle
        # is trained on
        "ring_length": [220, 270],
    },)

env_name = EnergyOptSPDEnv

flow_params = dict(
    # name of the experiment
    exp_tag=name,
    # name of the flow environment the experiment is running on
    env_name=env_name,
    # name of the network class the experiment uses
    network=network_name,
    # simulator that is used by the experiment
    simulator='traci',
    # simulation-related parameters
    sim=sim_params,
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and
    # the network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,
    # vehicles to be placed in the network at the start of a rollout
    # (see flow.core.vehicles.Vehicles)
    veh=vehicles,
    # (optional) parameters affecting the positioning of vehicles upon
    # initialization/reset (see flow.core.params.InitialConfig)
    initial=initial_config
)