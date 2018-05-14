import os
import urllib.request

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoLaneChangeParams, SumoCarFollowingParams, InFlows
from flow.core.vehicles import Vehicles

from flow.core.experiment import SumoExperiment
from flow.envs.bay_bridge import BridgeBaseEnv
from flow.scenarios.bottleneck.gen import BottleneckGenerator
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.controllers import SumoCarFollowingController, BayBridgeRouter

NETFILE = "bottleneck.net.xml"


def bay_bridge_bottleneck_example(sumo_binary=None,
                                  use_traffic_lights=False):
    """
    Performs a non-RL simulation of the bottleneck portion of the Oakland-San
    Francisco Bay Bridge. This consists of the toll booth and sections of the
    road leading up to it.
    Parameters
    ----------
    sumo_binary: bool, optional
        specifies whether to use sumo's gui during execution
    use_traffic_lights: bool, optional
        whether to activate the traffic lights in the scenario
    Note
    ----
    Unlike the bay_bridge_example, inflows are always activated here.
    """
    sumo_params = SumoParams(sim_step=0.6,
                             overtake_right=True)

    if sumo_binary is not None:
        sumo_params.sumo_binary = sumo_binary

    sumo_car_following_params = SumoCarFollowingParams(speedDev=0.2)
    sumo_lc_params = SumoLaneChangeParams(
            model="LC2013", lcCooperative=0.2, lcSpeedGain=15)

    vehicles = Vehicles()

    vehicles.add(veh_id="human",
                 acceleration_controller=(SumoCarFollowingController, {}),
                 routing_controller=(BayBridgeRouter, {}),
                 speed_mode="all_checks",
                 lane_change_mode="no_lat_collide",
                 sumo_car_following_params=sumo_car_following_params,
                 sumo_lc_params=sumo_lc_params,
                 num_vehicles=300)

    additional_env_params = {"target_velocity": 8}
    env_params = EnvParams(additional_params=additional_env_params)

    inflow = InFlows()

    inflow.add(veh_type="human", edge="393649534", probability=0.2,
               departLane="0", departSpeed=20)

    inflow.add(veh_type="human", edge="4757680", probability=0.2,
               departLane="0", departSpeed=20)

    inflow.add(veh_type="human", edge="32661316", probability=0.2,
               departLane="0", departSpeed=20)
    inflow.add(veh_type="human", edge="32661316", probability=0.2,
               departLane="1", departSpeed=20)

    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="0", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="1", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="2", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="3", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="4", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="5", departSpeed=20)
    inflow.add(veh_type="human", edge="90077193#0", probability=0.2,
               departLane="6", departSpeed=20)

    net_params = NetParams(in_flows=inflow,
                           no_internal_links=False)
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

    with open(os.path.join(net_params.cfg_path, NETFILE), "wb+") as f:
        f.write(data_to_write)

    initial_config = InitialConfig(spacing="uniform",  # "random",
                                   lanes_distribution=float("inf"),
                                   min_gap=15)

    scenario = BottleneckScenario(name="bottleneck",
                                  generator_class=BottleneckGenerator,
                                  vehicles=vehicles,
                                  net_params=net_params,
                                  initial_config=initial_config)

    env = BridgeBaseEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = bay_bridge_bottleneck_example(sumo_binary="sumo-gui",
                                        use_traffic_lights=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
