"""Bay Bridge toll example."""

import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoLaneChangeParams, SumoCarFollowingParams, InFlows
from flow.core.vehicles import Vehicles

from flow.core.experiment import SumoExperiment
from flow.envs.bay_bridge import BayBridgeEnv
from flow.scenarios.bay_bridge_toll.gen import BayBridgeTollGenerator
from flow.scenarios.bay_bridge_toll.scenario import BayBridgeTollScenario
from flow.controllers import SumoCarFollowingController, BayBridgeRouter

NETFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bottleneck.net.xml")


def bay_bridge_bottleneck_example(render=None, use_traffic_lights=False):
    """Perform a simulation of the toll portion of the Bay Bridge.

    This consists of the toll booth and sections of the road leading up to it.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use sumo's gui during execution
    use_traffic_lights: bool, optional
        whether to activate the traffic lights in the scenario

    Note
    ----
    Unlike the bay_bridge_example, inflows are always activated here.
    """
    sumo_params = SumoParams(sim_step=0.4, overtake_right=True)

    if render is not None:
        sumo_params.render = render

    sumo_car_following_params = SumoCarFollowingParams(
        speedDev=0.2,
        speed_mode="all_checks",
    )
    sumo_lc_params = SumoLaneChangeParams(
        model="LC2013",
        lcCooperative=0.2,
        lcSpeedGain=15,
        lane_change_mode="no_lat_collide",
    )

    vehicles = Vehicles()

    vehicles.add(
        veh_id="human",
        acceleration_controller=(SumoCarFollowingController, {}),
        routing_controller=(BayBridgeRouter, {}),
        sumo_car_following_params=sumo_car_following_params,
        sumo_lc_params=sumo_lc_params,
        num_vehicles=50)

    additional_env_params = {}
    env_params = EnvParams(additional_params=additional_env_params)

    inflow = InFlows()

    inflow.add(
        veh_type="human",
        edge="393649534",
        probability=0.2,
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="4757680",
        probability=0.2,
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="32661316",
        probability=0.2,
        departLane="random",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="90077193#0",
        vehs_per_hour=2000,
        departLane="random",
        departSpeed=10)

    net_params = NetParams(
        inflows=inflow, no_internal_links=False, netfile=NETFILE)

    # download the netfile from AWS
    if use_traffic_lights:
        my_url = "https://s3-us-west-1.amazonaws.com/flow.netfiles/" \
                 "bay_bridge_TL_all_green.net.xml"
    else:
        my_url = "https://s3-us-west-1.amazonaws.com/flow.netfiles/" \
                 "bay_bridge_junction_fix.net.xml"
    my_file = urllib.request.urlopen(my_url)
    data_to_write = my_file.read()

    with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), NETFILE),
            "wb+") as f:
        f.write(data_to_write)

    initial_config = InitialConfig(
        spacing="uniform",  # "random",
        min_gap=15)

    scenario = BayBridgeTollScenario(
        name="bay_bridge_toll",
        generator_class=BayBridgeTollGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = BayBridgeEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bay_bridge_bottleneck_example(
        render=True, use_traffic_lights=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
