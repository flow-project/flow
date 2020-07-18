"""Used as an example of sugiyama experiment.

This example consists of 10 IDM cars and a bus on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams, SumoCarFollowingParams, \
    BusStops
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.controllers.base_routing_controller import BaseRouter
import numpy as np


class ContinuousBusRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed ring.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.

    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """
        edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif edge == current_route[-1]:
            # choose one of the available routes based on the fraction of times
            # the given route can be chosen
            num_routes = len(env.available_bus_routes[edge])
            # import ipdb; ipdb.set_trace()
            frac = [val[1] for val in env.available_bus_routes[edge]]
            route_id = np.random.choice(
                [i for i in range(num_routes)], size=1, p=frac)[0]

            # pass the chosen route
            return 'bus_route{}_{}'.format(edge, route_id)
        else:
            return None


class RingNetwork(RingNetwork):
    # def specify_bus_routes(self, net_params):
    #     return [k['id'] for k, v in net_params.bus_stops.get().items()]

    def specify_bus_routes(self, net_params):
        """See parent class."""
        rts = {
            "top": (["top", "left", "bottom", "right"], ["bus_stop_0", "bus_stop_1"]),
            "left": (["left", "bottom", "right", "top"], ["bus_stop_0", "bus_stop_1"]),
            "bottom": (["bottom", "right", "top", "left"], ["bus_stop_0", "bus_stop_1"]),
            "right": (["right", "top", "left", "bottom"], ["bus_stop_0", "bus_stop_1"]),
        }

        return rts


def sugiyama_example(render=None):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sim_params = SumoParams(sim_step=0.1, render=render)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="bus",
        acceleration_controller=(IDMController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
            sigma=0,
            length=12,
            gui_shape="bus",
        ),
        routing_controller=(ContinuousBusRouter, {}),
        num_vehicles=1)
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=10)

    stops = BusStops()
    stops.add(
        edge="bottom",
        start_pos=0,
        end_pos=15
    )
    stops.add(
        edge="top",
        start_pos=0,
        end_pos=15
    )
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        additional_params=additional_net_params,
        bus_stops=stops)

    initial_config = InitialConfig(bunching=20)

    network = RingNetwork(
        name="sugiyama",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = AccelEnv(env_params, sim_params, network)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example(render=True)

    # run for a set number of rollouts / time steps
    exp.run(1, 1000)
