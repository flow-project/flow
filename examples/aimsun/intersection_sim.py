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
from flow.networks import SimpleIntNetwork

ADDITIONAL_NET_PARAMS = {
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # length of the four edges
    "edge_length": 100, 
    ### Specify if we want turns to be on
    "turns_on": True,
    # One way or Two Way?
    "one_way": False
}

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
    ### Change the following lines to change between aimsun and traci
    sim_params = SumoParams(sim_step=0.5, render=True, emission_path='data')
    simulator = 'traci'  #'traci' or 'aimsun'

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
        edge="left_center",
        vehs_per_hour=1000,
        departLane="free",
        departSpeed=0)

    inflow.add(
        veh_type="human",
        edge="bottom_center",
        vehs_per_hour=1000,
        departLane="free",
        departSpeed=0)

    if not ADDITIONAL_NET_PARAMS["one_way"]:
        inflow.add(
            veh_type="human",
            edge="right_center",
            vehs_per_hour=1000,
            departLane="free",
            departSpeed=0)

        inflow.add(
            veh_type="human",
            edge="top_center",
            vehs_per_hour=1000,
            departLane="free",
            departSpeed=0)

    net_params = NetParams(inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS.copy())

    network = SimpleIntNetwork(
        name="intersection",
        vehicles=vehicles,
        net_params=net_params)

    env = TestEnv(env_params, sim_params, network, simulator=simulator)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = simple_intersection_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
