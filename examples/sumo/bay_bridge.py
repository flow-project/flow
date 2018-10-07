"""Bay Bridge simulation."""

import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams, InFlows
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.core.experiment import SumoExperiment
from flow.envs.bay_bridge import BayBridgeEnv
from flow.scenarios.bay_bridge.gen import BayBridgeGenerator
from flow.scenarios.bay_bridge.scenario import BayBridgeScenario
from flow.controllers import SumoCarFollowingController, BayBridgeRouter

NETFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bay_bridge.net.xml")


def bay_bridge_example(render=None,
                       use_inflows=False,
                       use_traffic_lights=False):
    """
    Perform a simulation of vehicles on the Oakland-San Francisco Bay Bridge.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution
    use_inflows: bool, optional
        whether to activate inflows from the peripheries of the network
    use_traffic_lights: bool, optional
        whether to activate the traffic lights in the scenario

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles simulated by sumo on the Bay Bridge.
    """
    sumo_params = SumoParams(sim_step=0.6, overtake_right=True)

    if render is not None:
        sumo_params.render = render

    sumo_car_following_params = SumoCarFollowingParams(
        speedDev=0.2,
        speed_mode="all_checks",
    )
    sumo_lc_params = SumoLaneChangeParams(
        lcAssertive=20,
        lcPushy=0.8,
        lcSpeedGain=4.0,
        model="LC2013",
        lane_change_mode="no_lat_collide",
        # lcKeepRight=0.8
    )

    vehicles = Vehicles()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SumoCarFollowingController, {}),
        routing_controller=(BayBridgeRouter, {}),
        sumo_car_following_params=sumo_car_following_params,
        sumo_lc_params=sumo_lc_params,
        num_vehicles=1400)

    additional_env_params = {}
    env_params = EnvParams(additional_params=additional_env_params)

    traffic_lights = TrafficLights()

    inflow = InFlows()

    if use_inflows:
        # south
        inflow.add(
            veh_type="human",
            edge="183343422",
            vehsPerHour=528,
            departLane="0",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="183343422",
            vehsPerHour=864,
            departLane="1",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="183343422",
            vehsPerHour=600,
            departLane="2",
            departSpeed=20)

        inflow.add(
            veh_type="human",
            edge="393649534",
            probability=0.1,
            departLane="0",
            departSpeed=20)  # no data for this

        # west
        inflow.add(
            veh_type="human",
            edge="11189946",
            vehsPerHour=1752,
            departLane="0",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="11189946",
            vehsPerHour=2136,
            departLane="1",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="11189946",
            vehsPerHour=576,
            departLane="2",
            departSpeed=20)

        # north
        inflow.add(
            veh_type="human",
            edge="28413687#0",
            vehsPerHour=2880,
            departLane="0",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="28413687#0",
            vehsPerHour=2328,
            departLane="1",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="28413687#0",
            vehsPerHour=3060,
            departLane="2",
            departSpeed=20)
        inflow.add(
            veh_type="human",
            edge="11198593",
            probability=0.1,
            departLane="0",
            departSpeed=20)  # no data for this
        inflow.add(
            veh_type="human",
            edge="11197889",
            probability=0.1,
            departLane="0",
            departSpeed=20)  # no data for this

        # midway through bridge
        inflow.add(
            veh_type="human",
            edge="35536683",
            probability=0.1,
            departLane="0",
            departSpeed=20)  # no data for this

    net_params = NetParams(inflows=inflow, no_internal_links=False)
    net_params.netfile = NETFILE

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

    initial_config = InitialConfig(spacing="uniform", min_gap=15)

    scenario = BayBridgeScenario(
        name="bay_bridge",
        generator_class=BayBridgeGenerator,
        vehicles=vehicles,
        traffic_lights=traffic_lights,
        net_params=net_params,
        initial_config=initial_config)

    env = BayBridgeEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bay_bridge_example(
        render=True, use_inflows=False, use_traffic_lights=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
