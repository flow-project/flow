"""Ring road example.

Trains a number of autonomous vehicles to stabilize the flow of 22 vehicles in
a variable length ring road.
"""
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.controllers.lane_change_controllers import SafeAggressiveLaneChanger
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env


def make_flow_params(horizon, num_total_veh, num_av, num_lanes, ring_length, num_aggressive=0, bunching=0):

    vehicles = VehicleParams()
    # Add one automated vehicle.
    vehicles.add(
        veh_id="av",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_av,
        color='red')

    num_human = num_total_veh - num_av - num_aggressive

    vehicles.add(
        veh_id="human",
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="strategic",
        ),
        acceleration_controller=(IDMController, {"a": 0.3, "b": 2.0, "noise": 0.5}),
        car_following_params=SumoCarFollowingParams(speed_mode='no_collide'),
        # lane_change_controller=(SafeAggressiveLaneChanger, {"target_velocity": 8.0, "threshold": 1.0}),
        # lane_change_params=SumoLaneChangeParams(lcAssertive=0.5,lcSpeedGain=100, lcSpeedGainRight=1.0),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_human,
        initial_speed=5)

    vehicles.add(
        "aggressive",
        acceleration_controller=(IDMController, {"a": 3.0, "b": 8.0, "noise": 0.1, "T": 0.0, "s0": 0.0}),
        car_following_params=SumoCarFollowingParams(speed_mode='no_collide'),
        lane_change_controller=(SafeAggressiveLaneChanger, {
                                "target_velocity": 100.0, "threshold": 1.0, "desired_lc_time_headway": 0.1}),
        lane_change_params=SumoLaneChangeParams(lane_change_mode="no_lat_collide"),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=num_aggressive,
        color='green',
        initial_speed=5
    )

    flow_params = dict(
        # name of the experiment
        exp_tag="multilane_ring_smoothing",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentWaveAttenuationPOEnv,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=False,
            restart_instance=False,
            use_ballistic=True,
            disable_collisions=True
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=horizon,
            warmup_steps=500,
            clip_actions=False,
            additional_params={
                "max_accel": 2.6,
                "max_decel": 4.5,
                "ring_length": [ring_length, ring_length],
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params={
                "length": ring_length,
                "lanes": num_lanes,
                "speed_limit": 30,
                "resolution": 230,
            }, ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(spacing="uniform", shuffle=True, bunching=bunching),
    )

    return flow_params


flow_params = make_flow_params(4000, 22, 1, 1, 220)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}


# Setup PG with an ensemble of `num_policies` different policy graphs
POLICY_GRAPHS = {'av': gen_policy()}


def policy_mapping_fn(_):
    """Map a policy in RLlib."""
    return 'av'


POLICIES_TO_TRAIN = ['av']
