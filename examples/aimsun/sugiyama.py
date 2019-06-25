"""Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController
from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS


def sugiyama_example(render=None):
    """Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sim_params = AimsunParams(sim_step=0.5, render=True, emission_path='data')

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        num_vehicles=22)

    env_params = EnvParams()

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(bunching=20)

    scenario = LoopScenario(
        name="sugiyama",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = TestEnv(env_params, sim_params, scenario, simulator='aimsun')

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3000)
