"""UDSSC merge example."""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
     InFlows, SumoCarFollowingParams, SumoLaneChangeParams, VehicleParams

from flow.controllers import RLController, IDMController, ContinuousRouter, \
    SimLaneChangeController
from flow.scenarios.figure_eight import ADDITIONAL_NET_PARAMS

# Training settings
HORIZON = 500
SIM_STEP = 1
ITR = 101
N_ROLLOUTS = 40
exp_tag = "icra_15"  # experiment prefix

# # Local settings
# N_CPUS = 1
# RENDER = False
# MODE = "local"
# RESTART_INSTANCE = True
# LOCAL = True

# Autoscaler settings
N_CPUS = 10
RENDER = False
MODE = "local"
RESTART_INSTANCE = True
LOCAL = False

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()


# Inner ring vehicles
vehicles.add(veh_id="idm",
                acceleration_controller=(IDMController, {"noise": 0.1}),
                lane_change_controller=(SimLaneChangeController, {}),
                routing_controller=(ContinuousRouter, {}),
                num_vehicles=1,
                car_following_params=SumoCarFollowingParams(
                    accel=1,
                    decel=1, 
                    tau=1.1,
                    impatience=0.05,
                    max_speed=8,
                    speed_mode="all_checks",
                ),
                lane_change_params=SumoLaneChangeParams(
                    lane_change_mode=0,
                )
            )

# A single learning agent in the inner ring
vehicles.add(veh_id="rl",
                acceleration_controller=(RLController, {}),
                lane_change_controller=(SimLaneChangeController, {}),
                routing_controller=(ContinuousRouter, {}),
                num_vehicles=1,
                car_following_params=SumoCarFollowingParams(
                    tau=1.1,
                    impatience=0.05,
                    max_speed=8,
                    speed_mode="no_collide",
                ),
                lane_change_params=SumoLaneChangeParams(
                    lane_change_mode="aggressive"
                )
            )

inflow = InFlows()
    
inflow.add(veh_type="rl", edge="inflow_0", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_0", name="idm", vehs_per_hour=50)

inflow.add(veh_type="rl", edge="inflow_1", name="rl", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)
inflow.add(veh_type="idm", edge="inflow_1", name="idm", vehs_per_hour=50)


flow_params = dict(
    # name of the experiment
    exp_tag=exp_tag,

    # name of the flow environment the experiment is running on
    env_name='UDSSCMergeEnvReset',

    # name of the scenario class the experiment is running on
    scenario='UDSSCMergingScenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim = SumoParams(
        sim_step=SIM_STEP,
        render=RENDER,
        restart_instance=RESTART_INSTANCE
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params = {
            # maximum acceleration for autonomous vehicles, in m/s^2
            "max_accel": 1,
            # maximum deceleration for autonomous vehicles, in m/s^2
            "max_decel": 1,
            # desired velocity for all vehicles in the network, in m/s
            "target_velocity": 8,
            # number of observable vehicles preceding the rl vehicle
            "n_preceding": 1, # HAS TO BE 1
            # number of observable vehicles following the rl vehicle
            "n_following": 1, # HAS TO BE 1
            # number of observable merging-in vehicle from the larger loop
            "n_merging_in": 6,
            # batch size, for use in UDSSCMergeEnvReset
            "batch_size": HORIZON * N_ROLLOUTS,
            # # rl action noise
            # "rl_action_noise": 0.5,
            # # noise to add to the state space
            # "state_noise": 0.1,
            # what portion of the ramp the RL vehicle isn't controlled for 
            "control_length": 0.1,
            # range of inflow lengths for inflow_0, inclusive
            "range_inflow_0": [1, 4],
            # range of inflow lengths for inflow_1, inclusive
            "range_inflow_1": [1, 7],
        }
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params = {
            # radius of the loops
            "ring_radius": 15,#15.25,
            # length of the straight edges connected the outer loop to the inner loop
            "lane_length": 55,
            # length of the merge next to the roundabout
            "merge_length": 15,
            # number of lanes in the inner loop
            "inner_lanes": 1,
            # number of lanes in the outer loop
            "outer_lanes": 1,
            # max speed limit in the roundabout
            "roundabout_speed_limit": 10,
            # max speed limit in the rest of the roundabout
            "outside_speed_limit": 10,
            # resolution of the curved portions
            "resolution": 100,
            # num lanes
            "lane_num": 1,
        }
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial = InitialConfig(
        x0=50,
        spacing="custom", # TODO make this custom? 
        additional_params={"merge_bunching": 0}
    ),
)


def setup_exps():

    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [100, 50, 25]})
    config['sgd_minibatch_size'] = 128
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['horizon'] = HORIZON
    config['vf_loss_coeff'] = 1.0
    config['vf_clip_param'] = 10.0

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    if LOCAL:
        ray.init(num_cpus=N_CPUS+1, object_store_memory=10000000)
    else:
        ray.init(num_cpus=N_CPUS+1)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 20,
            "checkpoint_at_end": True,
            'max_failures': 999,
            'stop': {
                'training_iteration': ITR,
            },
            'upload_dir': 's3://kathy.experiments/rllib/experiments',
            'num_samples': 3,
        }
    })
