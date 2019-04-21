"""Example of a figure 8 network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""
from flow.controllers import IDMController
from flow.core.experiment import Experiment
from flow.core.params import AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.scenarios.figure_eight import Figure8Scenario, ADDITIONAL_NET_PARAMS


def figure_eight_example(render=None):
    """Perform a simulation of vehicles on a figure eight.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a figure eight.
    """
    sim_params = AimsunParams(sim_step=0.5, render=False, emission_path='data')

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {}),
        num_vehicles=14)

    env_params = EnvParams()

    net_params = NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy())

    scenario = Figure8Scenario(
        name="figure8",
        vehicles=vehicles,
        net_params=net_params)

    env = TestEnv(env_params, sim_params, scenario, simulator='aimsun')

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = figure_eight_example(render=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
