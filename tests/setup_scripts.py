import logging
from numpy import pi, sin, cos, linspace

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController

from flow.envs.loop_accel import AccelEnv

from flow.scenarios.loop.gen import CircleGenerator
from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.scenarios.figure8.gen import Figure8Generator
from flow.scenarios.figure8.figure8_scenario import Figure8Scenario


def ring_road_exp_setup(sumo_params=None,
                        vehicles=None,
                        env_params=None,
                        net_params=None,
                        initial_config=None,
                        traffic_lights=None):
    """
    Creates an environment and scenario pair for ring road test experiments.

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
        specifies starting positions of vehicles, defaults to evenly distributed
        vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1,
                                 sumo_binary="sumo")

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     speed_mode="aggressive",
                     num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {"target_velocity": 8, "num_steps": 500}
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig()

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = LoopScenario(name="RingRoadTest",
                            generator_class=CircleGenerator,
                            vehicles=vehicles,
                            net_params=net_params,
                            initial_config=initial_config,
                            traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(env_params=env_params,
                   sumo_params=sumo_params,
                   scenario=scenario)

    return env, scenario


def figure_eight_exp_setup(sumo_params=None,
                           vehicles=None,
                           env_params=None,
                           net_params=None,
                           initial_config=None,
                           traffic_lights=None):
    """
    Creates an environment and scenario pair for figure eight test experiments.

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
        specifies starting positions of vehicles, defaults to evenly distributed
        vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1,
                                 sumo_binary="sumo")

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(IDMController, {}),
                     speed_mode="aggressive",
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {"target_velocity": 8, "num_steps": 500}
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {"radius_ring": 30, "lanes": 1,
                                 "speed_limit": 30, "resolution": 40}
        net_params = NetParams(no_internal_links=False,
                               additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig()

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = Figure8Scenario(name="RingRoadTest",
                               generator_class=Figure8Generator,
                               vehicles=vehicles,
                               net_params=net_params,
                               initial_config=initial_config,
                               traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(env_params=env_params,
                   sumo_params=sumo_params,
                   scenario=scenario)

    return env, scenario


def variable_lanes_exp_setup(sumo_params=None,
                             vehicles=None,
                             env_params=None,
                             net_params=None,
                             initial_config=None,
                             traffic_lights=None):
    """
    Creates an environment and scenario pair for a ring road network with
    different number of lanes in each edge. Used for test purposes.

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
        specifies starting positions of vehicles, defaults to evenly distributed
        vehicles across the length of the network
    traffic_lights: TrafficLights type
        traffic light signals, defaults to no traffic lights in the network
    """
    logging.basicConfig(level=logging.WARNING)

    if sumo_params is None:
        # set default sumo_params configuration
        sumo_params = SumoParams(sim_step=0.1,
                                 sumo_binary="sumo")

    if vehicles is None:
        # set default vehicles configuration
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(IDMController, {}),
                     speed_mode="aggressive",
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=1)

    if env_params is None:
        # set default env_params configuration
        additional_env_params = {"target_velocity": 8, "num_steps": 500}
        env_params = EnvParams(additional_params=additional_env_params)

    if net_params is None:
        # set default net_params configuration
        additional_net_params = {"length": 230, "lanes": 1,
                                 "speed_limit": 30, "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

    if initial_config is None:
        # set default initial_config configuration
        initial_config = InitialConfig()

    if traffic_lights is None:
        # set default to no traffic lights
        traffic_lights = TrafficLights()

    # create the scenario
    scenario = LoopScenario(name="VariableLaneRingRoadTest",
                            generator_class=VariableLanesGenerator,
                            vehicles=vehicles,
                            net_params=net_params,
                            initial_config=initial_config,
                            traffic_lights=traffic_lights)

    # create the environment
    env = AccelEnv(env_params=env_params,
                   sumo_params=sumo_params,
                   scenario=scenario)

    return env, scenario


class VariableLanesGenerator(CircleGenerator):
    def specify_edges(self, net_params):
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        v = net_params.additional_params["speed_limit"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [
            {"id": "bottom", "from": "bottom", "to": "right", "speed": str(v),
             "length": repr(edgelen), "numLanes": "1",
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi / 2, 0, resolution)])},
            {"id": "right", "from": "right", "to": "top", "speed": str(v),
             "length": repr(edgelen), "numLanes": "3",
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(0, pi / 2, resolution)])},
            {"id": "top", "from": "top", "to": "left", "speed": str(v),
             "length": repr(edgelen), "numLanes": "2",
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(pi / 2, pi, resolution)])},
            {"id": "left", "from": "left", "to": "bottom", "speed": str(v),
             "length": repr(edgelen), "numLanes": "4",
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(pi, 3 * pi / 2, resolution)])}
        ]

        return edges
