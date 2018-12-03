"""Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.netfile import NetFileScenario


def figure_eight_example(render=None):
    """
    Perform a simulation of vehicles on a figure eight.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a figure eight.
    """
    sumo_params = SumoParams(render=False, sim_step=1, emission_path='data/', no_step_log=False)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()

    env_params = EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS)

    net_params = NetParams(
        no_internal_links=False)

    scenario = NetFileScenario(
        name="figure8",
        vehicles=vehicles,
        net_params=net_params)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = figure_eight_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3600*24, convert_to_csv=True)
