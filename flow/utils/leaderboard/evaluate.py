"""
Evaluation utility methods for testing the performance of controllers.

This file contains a method to perform the evaluation on all benchmarks in
flow/benchmarks, as well as method for importing neural network controllers
from rllib.
"""

from flow.core.experiment import Experiment
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.utils.rllib import get_flow_params, get_rllib_config
from flow.utils.registry import make_create_env
from flow.utils.exceptions import FatalFlowError

from flow.benchmarks.grid0 import flow_params as grid0
from flow.benchmarks.grid1 import flow_params as grid1
from flow.benchmarks.bottleneck0 import flow_params as bottleneck0
from flow.benchmarks.bottleneck1 import flow_params as bottleneck1
from flow.benchmarks.bottleneck2 import flow_params as bottleneck2
from flow.benchmarks.figureeight0 import flow_params as figureeight0
from flow.benchmarks.figureeight1 import flow_params as figureeight1
from flow.benchmarks.figureeight2 import flow_params as figureeight2
from flow.benchmarks.merge0 import flow_params as merge0
from flow.benchmarks.merge1 import flow_params as merge1
from flow.benchmarks.merge2 import flow_params as merge2

import ray
from ray.rllib.agent import get_agent_class
from ray.tune.registry import get_registry, register_env
import numpy as np

# number of simulations to execute when computing performance scores
NUM_RUNS = 10

# dictionary containing all available benchmarks and their meta-parameters
AVAILABLE_BENCHMARKS = {
    "grid0": grid0,
    "grid1": grid1,
    "bottleneck0": bottleneck0,
    "bottleneck1": bottleneck1,
    "bottleneck2": bottleneck2,
    "figureeight0": figureeight0,
    "figureeight1": figureeight1,
    "figureeight2": figureeight2,
    "merge0": merge0,
    "merge1": merge1,
    "merge2": merge2
}


def evaluate_policy(benchmark, _get_actions, _get_states=None):
    """Evaluate the performance of a controller on a predefined benchmark.

    Parameters
    ----------
    benchmark : str
        name of the benchmark, must be printed as it is in the
        benchmarks folder; otherwise a FatalFlowError will be raised
    _get_actions : method
        the mapping from states to actions for the RL agent(s)
    _get_states : method, optional
        a mapping from the environment object in Flow to some state, which
        overrides the _get_states method of the environment. Note that the
        same cannot be done for the actions.

    Returns
    -------
    float
        mean of the evaluation return of the benchmark from NUM_RUNS number
        of simulations
    float
        standard deviation of the evaluation return of the benchmark from
        NUM_RUNS number of simulations

    Raises
    ------
    flow.utils.exceptions.FatalFlowError
        If the specified benchmark is not available.
    """
    if benchmark not in AVAILABLE_BENCHMARKS.keys():
        raise FatalFlowError(
            "benchmark {} is not available. Check spelling?".format(benchmark))

    # get the flow params from the benchmark
    flow_params = AVAILABLE_BENCHMARKS[benchmark]

    exp_tag = flow_params["exp_tag"]
    sim_params = flow_params["sim"]
    vehicles = flow_params["veh"]
    env_params = flow_params["env"]
    env_params.evaluate = True  # Set to true to get evaluation returns
    net_params = flow_params["net"]
    initial_config = flow_params.get("initial", InitialConfig())
    traffic_lights = flow_params.get("tls", TrafficLightParams())

    # import the environment and network classes
    module = __import__("flow.envs", fromlist=[flow_params["env_name"]])
    env_class = getattr(module, flow_params["env_name"])
    module = __import__("flow.networks", fromlist=[flow_params["network"]])
    network_class = getattr(module, flow_params["network"])

    # recreate the network and environment
    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # make sure the _get_states method of the environment is the one
    # specified by the user
    if _get_states is not None:

        class _env_class(env_class):
            def get_state(self):
                return _get_states(self)

        env_class = _env_class

    env = env_class(
        env_params=env_params, sim_params=sim_params, network=network)

    flow_params = dict(
        # name of the experiment
        exp_tag=exp_tag,

        # name of the flow environment the experiment is running on
        env_name=env_class,

        # name of the network class the experiment is running on
        network=network_class,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=sim_params,

        # environment related parameters (see flow.core.params.EnvParams)
        env=env_params,

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=net_params,

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=initial_config,

        # traffic lights to be introduced to specific nodes (see
        # flow.core.params.TrafficLightParams)
        tls=traffic_lights,
    )

    # number of time steps
    flow_params['env'].horizon = env.env_params.horizon

    # create a Experiment object. Note that the state may not be that which is
    # specified by the environment.
    exp = Experiment(flow_params)
    exp.env = env

    exp = Experiment(flow_params)
    exp.env = env

    # run the experiment and return the reward
    res = exp.run(
        num_runs=NUM_RUNS,
        rl_actions=_get_actions)

    return np.mean(res["returns"]), np.std(res["returns"])


def get_compute_action_rllib(path_to_dir, checkpoint_num, alg):
    """Collect the compute_action method from RLlib's serialized files.

    Parameters
    ----------
    path_to_dir : str
        RLlib directory containing training results
    checkpoint_num : int
        checkpoint number / training iteration of the learned policy
    alg : str
        name of the RLlib algorithm that was used during the training
        procedure

    Returns
    -------
    method
        the compute_action method from the algorithm along with the trained
        parameters
    """
    # collect the configuration information from the RLlib checkpoint
    result_dir = path_to_dir if path_to_dir[-1] != '/' else path_to_dir[:-1]
    config = get_rllib_config(result_dir)

    # run on only one cpu for rendering purposes
    ray.init(num_cpus=1)
    config["num_workers"] = 1

    # create and register a gym+rllib env
    flow_params = get_flow_params(config)
    create_env, env_name = make_create_env(
        params=flow_params, version=9999, render=False)
    register_env(env_name, create_env)

    # recreate the agent
    agent_cls = get_agent_class(alg)
    agent = agent_cls(env=env_name, registry=get_registry(), config=config)

    # restore the trained parameters into the policy
    checkpoint = result_dir + '/checkpoint-{}'.format(checkpoint_num)
    agent._restore(checkpoint)

    return agent.compute_action
