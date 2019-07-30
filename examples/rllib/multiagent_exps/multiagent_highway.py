"""Ring road example.

Creates a set of stabilizing the ring experiments to test if
 more agents -> fewer needed batches
"""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams, SumoCarFollowingParams, InFlows, \
    SumoLaneChangeParams, VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.highway_ramps import HighwayRampsScenario, \
                                         ADDITIONAL_NET_PARAMS
from numpy import pi, sin, cos



# make sure (sample_batch_size * num_workers ~= train_batch_size)
# time horizon of a single rollout
HORIZON = 1000
# Number of rings
NUM_RINGS = 1
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 1
# number of iterations
TRAINING_ITERATIONS = 1



length_before = 150
length_between = 200
highway_inflow_rate = 2000
ramp_inflow_rate = 200

inflows = InFlows()
inflows.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=highway_inflow_rate,
    depart_lane="free",
    depart_speed="speedLimit",
    name="highway_inflow")
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=highway_inflow_rate // 2,
    depart_lane="free",
    depart_speed="speedLimit",
    name="highway_inflow")
inflows.add(
    veh_type="idm",
    edge="on_ramp_0",
    vehs_per_hour=ramp_inflow_rate,
    depart_lane="free",
    depart_speed="speedLimit",
    name="on_ramp_inflow")


# We place one autonomous vehicle and 21 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0,
        speed_mode="obey_safe_speed"
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0,
        speed_mode='obey_safe_speed',
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))

flow_params = dict(
    # name of the experiment
    exp_tag='multienv_pi',

    # name of the flow environment the experiment is running on
    env_name='MultiWaveAttenuationPOEnv',

    # name of the scenario class the experiment is running on
    scenario='HighwayRampsScenario',

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params={
            'max_accel': 1,
            'max_decel': 1,
            'target_velocity': 30
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params={
            # lengths of highway, on-ramps and off-ramps respectively
            "highway_length": 1500,
            "on_ramps_length": 200,
            "off_ramps_length": 200,
            # number of lanes on highway, on-ramps and off-ramps respectively
            "highway_lanes": 3,
            "on_ramps_lanes": 2,
            "off_ramps_lanes": 2,
            # speed limit on highway, on-ramps and off-ramps respectively
            "highway_speed": 30,
            "on_ramps_speed": 20,
            "off_ramps_speed": 20,
            # positions of the on-ramps
            "on_ramps_pos": [500],
            # positions of the off-ramps
            "off_ramps_pos": [1000],
            # probability for a vehicle to exit the highway at the next off-ramp
            "next_off_ramp_proba": 0.2,
            # ramps angles
            "angle_on_ramps": - 3 * pi / 4,
            "angle_off_ramps": - pi / 4
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['simple_optimizer'] = True
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['lr'] = tune.grid_search([1e-5])
    config['horizon'] = HORIZON
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': gen_policy()}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policy_graphs': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


if __name__ == '__main__':
    alg_run, env_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 5,
            'checkpoint_at_end': True,
            'stop': {
                'training_iteration': TRAINING_ITERATIONS
            },
            'config': config,
            # 'upload_dir': 's3://<BUCKET NAME>'
        },
    })
