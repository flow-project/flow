"""Specifies several methods for creating scenarios and environments.

This allows us to reduce the number of times these features are specified when
creating new tests, as all tests follow approximately the same format.
"""

import logging

from numpy import pi, sin, cos, linspace

from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter, GridRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.envs.green_wave_env import GreenWaveTestEnv
from flow.envs.loop.loop_accel import AccelEnv
from flow.scenarios.figure_eight import Figure8Scenario
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios.highway import HighwayScenario
from flow.scenarios.loop import LoopScenario


def ring_road_exp_setup(sim_params=None,
                        vehicles=None,
                        env_params=None,
                        net_params=None,
                        initial_config=None,
                        traffic_lights=None):
    """
    Create an environment and scenario pair for ring road test experiments.

    Parameters
    ----------
    sim_params : flow.core.params.SumoParams
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles : Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params : flow.core.params.EnvParams
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params : flow.core.params.NetParams
        network-specific configuration parameters, defaults to a single lane
        ring road of length 230 m
    initial_config : flow.core.params.InitialConfig
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights : flow.core.params.TrafficLightParams
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sim_params is None:
        # set default sim_params configuration
        sim_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {
            "target_velocity": 8,
            "max_accel": 1,
            "max_decel": 1,
            "sort_vehicles": False,
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
        traffic_lights = TrafficLightParams()

    # create the scenario
    scenario = LoopScenario(
        name="RingRoadTest",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sim_params=sim_params, scenario=scenario)

    # reset the environment
    env.reset()

    return env, scenario


def figure_eight_exp_setup(sim_params=None,
                           vehicles=None,
                           env_params=None,
                           net_params=None,
                           initial_config=None,
                           traffic_lights=None):
    """
    Create an environment and scenario pair for figure eight test experiments.

    Parameters
    ----------
    sim_params : flow.core.params.SumoParams
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles : Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params : flow.core.params.EnvParams
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params : flow.core.params.NetParams
        network-specific configuration parameters, defaults to a figure eight
        with a 30 m radius and "no_internal_links" set to False
    initial_config : flow.core.params.InitialConfig
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: flow.core.params.TrafficLightParams
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sim_params is None:
        # set default sim_params configuration
        sim_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
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
            "sort_vehicles": False
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
        traffic_lights = TrafficLightParams()

    # create the scenario
    scenario = Figure8Scenario(
        name="FigureEightTest",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sim_params=sim_params, scenario=scenario)

    # reset the environment
    env.reset()

    return env, scenario


def highway_exp_setup(sim_params=None,
                      vehicles=None,
                      env_params=None,
                      net_params=None,
                      initial_config=None,
                      traffic_lights=None):
    """
    Create an environment and scenario pair for highway test experiments.

    Parameters
    ----------
    sim_params : flow.core.params.SumoParams
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles : Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params : flow.core.params.EnvParams
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params : flow.core.params.NetParams
        network-specific configuration parameters, defaults to a single lane
        highway of length 100 m
    initial_config : flow.core.params.InitialConfig
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: flow.core.params.TrafficLightParams
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sim_params is None:
        # set default sim_params configuration
        sim_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
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
            "sort_vehicles": False
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
        traffic_lights = TrafficLightParams()

    # create the scenario
    scenario = HighwayScenario(
        name="RingRoadTest",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sim_params=sim_params, scenario=scenario)

    # reset the environment
    env.reset()

    return env, scenario


def grid_mxn_exp_setup(row_num=1,
                       col_num=1,
                       sim_params=None,
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
    sim_params : flow.core.params.SumoParams
        sumo-related configuration parameters, defaults to a time step of 1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles : Vehicles type
        vehicles to be placed in the network, default is 5 vehicles per edge
        for a total of 20 vehicles with an IDM acceleration controller and
        GridRouter routing controller.
    env_params : flow.core.params.EnvParams
        environment-specific parameters, defaults to a environment with
        failsafes, where other parameters do not matter for non-rl runs
    net_params : flow.core.params.NetParams
        network-specific configuration parameters, defaults to a 1x1 grid
        which traffic lights on and "no_internal_links" set to False
    initial_config : flow.core.params.InitialConfig
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    tl_logic: flow.core.params.TrafficLightParams
        specifies logic of any traffic lights added to the system
    """
    logging.basicConfig(level=logging.WARNING)

    if tl_logic is None:
        tl_logic = TrafficLightParams(baseline=False)

    if sim_params is None:
        # set default sim_params configuration
        sim_params = SumoParams(sim_step=1, render=False)

    if vehicles is None:
        vehicles_per_edge = 5
        num_edges = 2 * (row_num + col_num)
        total_vehicles = num_edges * vehicles_per_edge
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
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
        num_entries = 2 * row_num + 2 * col_num
        assert total_vehicles % num_entries == 0, "{} total vehicles should " \
                                                  "be divisible by {" \
                                                  "}".format(total_vehicles,
                                                             num_entries)
        grid_array = {
            "short_length": 100,
            "inner_length": 300,
            "long_length": 3000,
            "row_num": row_num,
            "col_num": col_num,
            "cars_left": int(total_vehicles / num_entries),
            "cars_right": int(total_vehicles / num_entries),
            "cars_top": int(total_vehicles / num_entries),
            "cars_bot": int(total_vehicles / num_entries)
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
            spacing="custom", additional_params={"enter_speed": 30})

    # create the scenario
    scenario = SimpleGridScenario(
        name="Grid1x1Test",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    # create the environment
    env = GreenWaveTestEnv(
        env_params=env_params, sim_params=sim_params, scenario=scenario)

    # reset the environment
    env.reset()

    return env, scenario


def variable_lanes_exp_setup(sim_params=None,
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
    sim_params : flow.core.params.SumoParams
        sumo-related configuration parameters, defaults to a time step of 0.1s
        and no sumo-imposed failsafe on human or rl vehicles
    vehicles : Vehicles type
        vehicles to be placed in the network, default is one vehicles with an
        IDM acceleration controller and ContinuousRouter routing controller.
    env_params : flow.core.params.EnvParams
        environment-specific parameters, defaults to a environment with no
        failsafes, where other parameters do not matter for non-rl runs
    net_params : flow.core.params.NetParams
        network-specific configuration parameters, defaults to a figure eight
        with a 30 m radius and "no_internal_links" set to False
    initial_config : flow.core.params.InitialConfig
        specifies starting positions of vehicles, defaults to evenly
        distributed vehicles across the length of the network
    traffic_lights: flow.core.params.TrafficLightParams
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sim_params is None:
        # set default sim_params configuration
        sim_params = SumoParams(sim_step=0.1, render=False)

    if vehicles is None:
        # set default vehicles configuration
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            car_following_params=SumoCarFollowingParams(
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
            "sort_vehicles": False
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
        traffic_lights = TrafficLightParams()

    # create the scenario
    scenario = VariableLanesScenario(
        name="VariableLaneRingRoadTest",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(
        env_params=env_params, sim_params=sim_params, scenario=scenario)

    # reset the environment
    env.reset()

    return env, scenario


class VariableLanesScenario(LoopScenario):
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
            v,
            "length":
            edgelen,
            "numLanes":
            1,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(-pi / 2, 0, resolution)
            ]
        }, {
            "id":
            "right",
            "from":
            "right",
            "to":
            "top",
            "speed":
            v,
            "length":
            edgelen,
            "numLanes":
            3,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(0, pi / 2, resolution)
            ]
        }, {
            "id":
            "top",
            "from":
            "top",
            "to":
            "left",
            "speed":
            v,
            "length":
            edgelen,
            "numLanes":
            2,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(pi / 2, pi, resolution)
            ]
        }, {
            "id":
            "left",
            "from":
            "left",
            "to":
            "bottom",
            "speed":
            v,
            "length":
            edgelen,
            "numLanes":
            4,
            "shape":
            [
                (r * cos(t), r * sin(t))
                for t in linspace(pi, 3 * pi / 2, resolution)
            ]
        }]

        return edges
