"""Example of a simple intersection network with human-driven vehicles.

Right-of-way dynamics near the intersection causes vehicles to queue up on
either side of the intersection, leading to a significant reduction in the
average speed of vehicles in the network.
"""

from flow.controllers import IDMController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, AimsunParams, EnvParams, NetParams
from flow.core.params import VehicleParams, InFlows
from flow.envs import TestEnv
from flow.scenarios.intersection import SimpleIntScenario, ADDITIONAL_NET_PARAMS


def simple_intersection_example(render=None):
    """Perform a simulation of vehicles on an intersection.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on an intersection.
    """
    sim_params = AimsunParams(sim_step=0.5, render=True, emission_path='data')

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {}),
        num_vehicles=0)

    env_params = EnvParams()

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="left",
        vehs_per_hour=1000,
        departLane="free",
        departSpeed=0)

    inflow.add(
        veh_type="human",
        edge="bottom",
        vehs_per_hour=1000,
        departLane="free",
        departSpeed=0)

    net_params = NetParams(inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS.copy())

    scenario = SimpleIntScenario(
        name="intersection",
        vehicles=vehicles,
        net_params=net_params)

    env = TestEnv(env_params, sim_params, scenario, simulator='aimsun')

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = simple_intersection_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
