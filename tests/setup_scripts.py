"""Specifies several methods for creating scenarios and environments.

This allows us to reduce the number of times these features are specified when
creating new tests, as all tests follow approximately the same format.
"""

import logging

from numpy import pi, sin, cos, linspace

from flow.controllers.car_following_models import IDMController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter, GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles
from flow.envs.green_wave_env import GreenWaveTestEnv
from flow.envs.loop.loop_accel import AccelEnv
from flow.scenarios.bottleneck.gen import BottleneckGenerator
from flow.scenarios.bottleneck.scenario import BottleneckScenario
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.scenarios.grid.grid_scenario import SimpleGridScenario
from flow.scenarios.highway.gen import HighwayGenerator
from flow.scenarios.highway.scenario import HighwayScenario
from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario


def ring_road_exp_setup(sumo_params=None,
                        vehicles=None,
                        env_params=None,
                        net_params=None,
                        initial_config=None,
                        traffic_lights=None):
    """
    Create an environment and scenario pair for ring road test experiments.

    Parameters
    ----------
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a single lane
        ring road of length 230 m
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "num_steps": 500
        }
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig(lanes_distribution=1)

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = LoopScenario(
        name="RingRoadTest",
        generator_class=CircleGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


def figure_eight_exp_setup(sumo_params=None,
                           vehicles=None,
                           env_params=None,
                           net_params=None,
                           initial_config=None,
                           traffic_lights=None):
    """
    Create an environment and scenario pair for figure eight test experiments.

    Parameters
    ----------
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a figure eight
        with a 30 m radius and "no_internal_links" set to False
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "num_steps": 500
        }
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {
            "radius_ring": 30,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig(lanes_distribution=1)

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = Figure8Scenario(
        name="RingRoadTest",
        generator_class=Figure8Generator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


def highway_exp_setup(sumo_params=None,
                      vehicles=None,
                      env_params=None,
                      net_params=None,
                      initial_config=None,
                      traffic_lights=None):
    """
    Create an environment and scenario pair for highway test experiments.

    Parameters
    ----------
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a single lane
        highway of length 100 m
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "num_steps": 500
        }
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {
            "length": 100,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
            "num_edges": 1
        }
        net_params = NetParams(additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig()

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = HighwayScenario(
        name="RingRoadTest",
        generator_class=HighwayGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


def grid_mxn_exp_setup(row_num=1,
                       col_num=1,
                       sumo_params=None,
                       vehicles=None,
                       env_params=None,
                       net_params=None,
                       initial_config=None,
                       tl_logic=None):
    """
    Create an environment and scenario pair for grid 1x1 test experiments.

    Parameters
    ----------
    row_num: int, optional
        number of horizontal rows of edges in the grid network
    col_num: int, optional
        number of vertical columns of edges in the grid network
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is 5 vehicles per edge
        for a total of 20 vehicles with an IDM acceleration controller and
        GridRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a 1x1 grid
        which traffic lights on and "no_internal_links" set to False
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    tl_logic: TrafficLights type
        specifies logic of any traffic lights added to the system
    """
    logging.basicConfig(level=logging.WARNING)

    if tl_logic is None:
        tl_logic = TrafficLights(baseline=False)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=1, render=False)

    if vehicles is None:
        total_vehicles = 20
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                min_gap=2.5, tau=1.1, max_speed=30),
            routing_controller=(GridRouter, {}),
            num_vehicles=total_vehicles)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 50,
            "switch_time": 3.0,
            "tl_type": "controlled",
            "discrete": False
        }

        env_params = EnvParams(
            additional_params=additional_env_params, horizon=100)

    if net_params is None:
        # set default net_params configuration
        total_vehicles = vehicles.num_vehicles
        grid_array = {
            "short_length": 100,
            "inner_length": 300,
            "long_length": 3000,
            "row_num": row_num,
            "col_num": col_num,
            "cars_left": int(total_vehicles / 4),
            "cars_right": int(total_vehicles / 4),
            "cars_top": int(total_vehicles / 4),
            "cars_bot": int(total_vehicles / 4)
        }

        additional_net_params = {
            "length": 200,
            "lanes": 2,
            "speed_limit": 35,
            "resolution": 40,
            "grid_array": grid_array,
            "horizontal_lanes": 1,
            "vertical_lanes": 1
        }

        net_params = NetParams(
            no_internal_links=False, additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig(
            spacing="uniform", additional_params={"enter_speed": 30})

    # create the scenario
    scenario = SimpleGridScenario(
        name="Grid1x1Test",
        generator_class=SimpleGridGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    # create the environment
    env = GreenWaveTestEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


def variable_lanes_exp_setup(sumo_params=None,
                             vehicles=None,
                             env_params=None,
                             net_params=None,
                             initial_config=None,
                             traffic_lights=None):
    """
    Create an environment and scenario variable-lane ring road.

    Each edge in this scenario can have a different number of lanes. Used for
    test purposes.

    Parameters
    ----------
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a figure eight
        with a 30 m radius and "no_internal_links" set to False
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "num_steps": 500
        }
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {
            "length": 230,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        }
        net_params = NetParams(additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig()

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = LoopScenario(
        name="VariableLaneRingRoadTest",
        generator_class=VariableLanesGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


def setup_bottlenecks(sumo_params=None,
                      vehicles=None,
                      env_params=None,
                      net_params=None,
                      initial_config=None,
                      traffic_lights=None,
                      inflow=None,
                      scaling=1):
    """
    Create an environment and scenario pair for grid 1x1 test experiments.

    Sumo-related configuration parameters, defaults to a time step of 1s
    and no sumo-imposed failsafe on human or rl vehicles

    Parameters
    ----------
    sumo_params: SumoParams type
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles: Vehicles type
        vehicles to be placed in the network, default is 5 vehicles per edge
        for a total of 20 vehicles with an IDM acceleration controller and
        GridRouter routing controller.
    env_params: EnvParams type
        environment-specific parameters, defaults to a environment with
        failsafes, where other parameters do not matter for non-rl runs
    net_params: NetParams type
        network-specific configuration parameters, defaults to a 1x1 grid
        which traffic lights on and "no_internal_links" set to False
    initial_config: InitialConfig type
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: TrafficLights type
        specifies logic of any traffic lights added to the system
    """
    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        vehicles = Vehicles()

        vehicles.add(
            veh_id="human",
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode=25,
            ),
            lane_change_controller=(SumoLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            sumo_lc_params=SumoLaneChangeParams(
                lane_change_mode=1621,
            ),
            num_vehicles=1 * scaling)

    if env_params is None:
        additional_env_params = {
            "target_velocity": 40,
            "max_accel": 1,
            "max_decel": 1,
            "lane_change_duration": 5,
            "add_rl_if_exit": False,
            "disable_tb": True,
            "disable_ramp_metering": True
        }
        env_params = EnvParams(additional_params=additional_env_params)

    if inflow is None:
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="1",
            vehsPerHour=1000,
            departLane="random",
            departSpeed=10)

    if traffic_lights is None:
        traffic_lights = TrafficLights()

    if net_params is None:
        additional_net_params = {"scaling": scaling}
        net_params = NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params)

    if initial_config is None:
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

    # create the environment
    env = AccelEnv(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario)

    return env, scenario


class VariableLanesGenerator(CircleGenerator):
    """Instantiate a ring road with variable number of lanes per edge."""

    def specify_edges(self, net_params):
        """See parent class.

        Each edge can be provided with a separate number of lanes.
        """
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        v = net_params.additional_params["speed_limit"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [{
            "id":
            "bottom",
            "from":
            "bottom",
            "to":
            "right",
            "speed":
            str(v),
            "length":
            repr(edgelen),
            "numLanes":
            "1",
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, 0, resolution)
            ])
        }, {
            "id":
            "right",
            "from":
            "right",
            "to":
            "top",
            "speed":
            str(v),
            "length":
            repr(edgelen),
            "numLanes":
            "3",
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(0, pi / 2, resolution)
            ])
        }, {
            "id":
            "top",
            "from":
            "top",
            "to":
            "left",
            "speed":
            str(v),
            "length":
            repr(edgelen),
            "numLanes":
            "2",
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(pi / 2, pi, resolution)
            ])
        }, {
            "id":
            "left",
            "from":
            "left",
            "to":
            "bottom",
            "speed":
            str(v),
            "length":
            repr(edgelen),
            "numLanes":
            "4",
            "shape":
            " ".join([
                "%.2f,%.2f" % (r * cos(t), r * sin(t))
                for t in linspace(pi, 3 * pi / 2, resolution)
            ])
        }]

        return edges
