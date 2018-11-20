"""Sugiyama experiment on many separate rings.

This example consists of 22 IDM cars on a ring creating shockwaves on
non overlapping rings used to test mult-iagent shared models.
"""

from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.core.experiment import SumoExperiment
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS
from flow.scenarios.lord_of_the_rings import ADDITIONAL_NET_PARAMS
from flow.scenarios.lord_of_the_rings import MultiLoopScenario


NUM_RINGS = 3


def sugiyama_example(render=None):
    """Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sumo_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    for i in range(NUM_RINGS):
        vehicles.add(
            veh_id='idm_{}'.format(i),
            routing_controller=(ContinuousRouter, {}),
            acceleration_controller=(IDMController, {}),
            num_vehicles=22)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.deepcopy()
    additional_net_params['num_rings'] = NUM_RINGS
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=20.0, spacing='custom')

    scenario = MultiLoopScenario(
        name='sugiyama',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == '__main__':
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
