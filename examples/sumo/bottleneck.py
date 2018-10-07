"""File demonstrating formation of congestion in bottleneck."""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.scenarios.bottleneck.gen import BottleneckGenerator
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.controllers import SumoLaneChangeController, ContinuousRouter
from flow.envs.bottleneck_env import BottleneckEnv
from flow.core.experiment import SumoExperiment

SCALING = 1
DISABLE_TB = True
# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 1800


def bottleneck_example(flow_rate, horizon, render=None):
    """
    Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottlneck
    horizon : int
        time horizon
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    if render is None:
        render = False

    sumo_params = SumoParams(
        sim_step=0.5,
        render=render,
        overtake_right=False,
        restart_instance=True)

    vehicles = Vehicles()

    vehicles.add(
        veh_id="human",
        lane_change_controller=(SumoLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        sumo_car_following_params=SumoCarFollowingParams(
            speed_mode=25,
        ),
        sumo_lc_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
        num_vehicles=1)

    additional_env_params = {
        "target_velocity": 40,
        "max_accel": 1,
        "max_decel": 1,
        "lane_change_duration": 5,
        "add_rl_if_exit": False,
        "disable_tb": DISABLE_TB,
        "disable_ramp_metering": DISABLE_RAMP_METER
    }
    env_params = EnvParams(
        horizon=horizon, additional_params=additional_env_params)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=flow_rate,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLights()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    additional_net_params = {"scaling": SCALING}
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"])

    scenario = BottleneckScenario(
        name="bay_bridge_toll",
        generator_class=BottleneckGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    env = BottleneckEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    # inflow, number of steps, binary
    exp = bottleneck_example(INFLOW, 1000, render=True)
    exp.run(5, 1000)
