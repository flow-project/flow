"""Multi-agent highway with ramps example.

Trains a non-constant number of agents, all sharing the same policy, on the
highway with ramps network.
"""
import json
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

from flow.controllers import RLController
from flow.core.params import EnvParams, NetParams, InitialConfig, InFlows, \
                             VehicleParams, SumoParams, \
                             SumoCarFollowingParams, SumoLaneChangeParams

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway_ramps import ADDITIONAL_NET_PARAMS


# SET UP PARAMETERS FOR THE SIMULATION

# number of training iterations
N_TRAINING_ITERATIONS = 200
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of steps per rollout
HORIZON = 1500
# number of parallel workers
N_CPUS = 11

# inflow rate on the highway in vehicles per hour
HIGHWAY_INFLOW_RATE = 4000
# inflow rate on each on-ramp in vehicles per hour
ON_RAMPS_INFLOW_RATE = 450
# percentage of autonomous vehicles compared to human vehicles on highway
PENETRATION_RATE = 20


# SET UP PARAMETERS FOR THE NETWORK

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # lengths of highway, on-ramps and off-ramps respectively
    "highway_length": 1500,
    "on_ramps_length": 250,
    "off_ramps_length": 250,
    # number of lanes on highway, on-ramps and off-ramps respectively
    "highway_lanes": 3,
    "on_ramps_lanes": 1,
    "off_ramps_lanes": 1,
    # speed limit on highway, on-ramps and off-ramps respectively
    "highway_speed": 30,
    "on_ramps_speed": 20,
    "off_ramps_speed": 20,
    # positions of the on-ramps
    "on_ramps_pos": [500],
    # positions of the off-ramps
    "off_ramps_pos": [1000],
    # probability for a vehicle to exit the highway at the next off-ramp
    "next_off_ramp_proba": 0.25
})


# SET UP PARAMETERS FOR THE ENVIRONMENT

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 1,
    'max_decel': 1,
    'target_velocity': 30
})


# CREATE VEHICLE TYPES AND INFLOWS

vehicles = VehicleParams()
inflows = InFlows()

# human vehicles
vehicles.add(
    veh_id="idm",
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",  # for safer behavior at the merges
        tau=1.5  # larger distance between cars
    ),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1621))

# autonomous vehicles
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}))

# add human vehicles on the highway
inflows.add(
    veh_type="idm",
    edge="highway_0",
    vehs_per_hour=HIGHWAY_INFLOW_RATE,
    depart_lane="free",
    depart_speed="max",
    name="idm_highway_inflow")

# add autonomous vehicles on the highway
# they will stay on the highway, i.e. they won't exit through the off-ramps
inflows.add(
    veh_type="rl",
    edge="highway_0",
    vehs_per_hour=int(HIGHWAY_INFLOW_RATE * PENETRATION_RATE / 100),
    depart_lane="free",
    depart_speed="max",
    name="rl_highway_inflow",
    route="routehighway_0_0")

# add human vehicles on all the on-ramps
for i in range(len(additional_net_params['on_ramps_pos'])):
    inflows.add(
        veh_type="idm",
        edge="on_ramp_{}".format(i),
        vehs_per_hour=ON_RAMPS_INFLOW_RATE,
        depart_lane="free",
        depart_speed="max",
        name="idm_on_ramp_inflow")


# SET UP FLOW PARAMETERS

flow_params = dict(
    exp_tag='multiagent_highway',
    env_name='MultiAgentHighwayPOEnv',
    network='HighwayRampsNetwork',
    simulator='traci',

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=200,
        sims_per_step=1,  # do not put more than one
        additional_params=additional_env_params,
    ),

    sim=SumoParams(
        sim_step=0.2,
        render=False,
        restart_instance=True
    ),

    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params),

    veh=vehicles,
    initial=InitialConfig(),
)


# SET UP EXPERIMENT

def setup_exps(flow_params):
    """Create the relevant components of a multiagent RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        input flow-parameters

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
    config['clip_actions'] = False
    config['observation_filter'] = 'NoFilter'

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # register as rllib env
    register_env(env_name, create_env)

    # multiagent configuration
    temp_env = create_env()
    policy_graphs = {'av': (PPOTFPolicy,
                            temp_env.observation_space,
                            temp_env.action_space,
                            {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            'policies_to_train': ['av']
        }
    })

    return alg_run, env_name, config


# RUN EXPERIMENT

if __name__ == '__main__':
    alg_run, env_name, config = setup_exps(flow_params)
    ray.init(num_cpus=N_CPUS + 1)

    run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': env_name,
            'checkpoint_freq': 20,
            'checkpoint_at_end': True,
            'stop': {
                'training_iteration': N_TRAINING_ITERATIONS
            },
            'config': config,
        },
    })
