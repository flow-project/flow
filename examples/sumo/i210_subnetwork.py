"""I-210 subnetwork example."""

import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoLaneChangeParams, SumoCarFollowingParams, InFlows
from flow.core.params import VehicleParams

from flow.core.experiment import Experiment
from flow.envs.test import TestEnv

from flow.networks.base import Network
from flow.networks.bay_bridge_toll import EDGES_DISTRIBUTION
from flow.controllers import SimCarFollowingController, BayBridgeRouter

NET_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../networks/sumo_i_210/test.net.xml")
ROUTE_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../networks/sumo_i_210/od_route_file.odtrips.rou.xml")


def i_210_example(render=None):
    """Perform a simulation of the toll portion of the Bay Bridge.

    This consists of the toll booth and sections of the road leading up to it.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Note
    ----
    Unlike the bay_bridge_example, inflows are always activated here.
    """
    sim_params = SumoParams(sim_step=0.4, overtake_right=True)

    if render is not None:
        sim_params.render = render

    car_following_params = SumoCarFollowingParams(
        speedDev=0.2,
        speed_mode="all_checks",
    )
    lane_change_params = SumoLaneChangeParams(
        model="LC2013",
        lcCooperative=0.2,
        lcSpeedGain=15,
        lane_change_mode="no_lat_collide",
    )

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        # routing_controller=(BayBridgeRouter, {}),
        car_following_params=car_following_params,
        lane_change_params=lane_change_params,
        num_vehicles=50)

    additional_env_params = {}
    env_params = EnvParams(additional_params=additional_env_params)

    # inflow = InFlows()
    #
    # inflow.add(
    #     veh_type="human",
    #     edge="393649534",
    #     probability=0.2,
    #     departLane="random",
    #     departSpeed=10)
    # inflow.add(
    #     veh_type="human",
    #     edge="4757680",
    #     probability=0.2,
    #     departLane="random",
    #     departSpeed=10)
    # inflow.add(
    #     veh_type="human",
    #     edge="32661316",
    #     probability=0.2,
    #     departLane="random",
    #     departSpeed=10)
    # inflow.add(
    #     veh_type="human",
    #     edge="90077193#0",
    #     vehs_per_hour=2000,
    #     departLane="random",
    #     departSpeed=10)

    # net_params = NetParams(inflows=inflow, template=TEMPLATE)
    net_params = NetParams(template={
                               # network geometry features
                               "net": NET_TEMPLATE,
                               # features associated with the routes vehicles take
                               "rou": ROUTE_TEMPLATE
                           }
                           )


    initial_config = InitialConfig(
        spacing="uniform",  # "random",
        min_gap=15)
        # edges_distribution=EDGES_DISTRIBUTION.copy())

    network = Network(
        name="i_210_test",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = TestEnv(env_params, sim_params, network)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = i_210_example(render=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
