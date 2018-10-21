"""Contains the ring road scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curves on the ring
    "resolution": 40,
    # number of rings in the system
    "num_rings": 7
}

VEHICLE_LENGTH = 5  # length of vehicles in the network, in meters


class MultiLoopScenario(Scenario):
    """Ring road scenario."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a loop scenario.

        Requires from net_params:
        - length: length of the circle
        - lanes: number of lanes in the circle
        - speed_limit: max speed limit of the circle
        - resolution: number of nodes resolution

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.num_rings = net_params.additional_params["num_rings"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        edgelen = self.length / 4
        shift = 4*edgelen

        edgestarts = []
        for i in range(self.num_rings):
            edgestarts += [("bottom_{}".format(i), 0 + i * shift),
                           ("right_{}".format(i), edgelen + i * shift),
                           ("top_{}".format(i), 2 * edgelen + i * shift),
                           ("left_{}".format(i), 3 * edgelen + i * shift)]

        return edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate uniformly spaced starting positions.

        It is assumed that there are an equal number of vehicles per ring.
        If the perturbation term in initial_config is set to some positive
        value, then the start positions are perturbed from a uniformly spaced
        distribution by a gaussian whose std is equal to this perturbation
        term.

        Parameters
        ----------
        initial_config : InitialConfig type
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network
        kwargs : dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions : list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
            list of start lanes
        """
        (x0, min_gap, bunching, lanes_distr, available_length,
         available_edges, initial_config) = \
            self._get_start_pos_util(initial_config, num_vehicles, **kwargs)

        increment = available_length / num_vehicles
        vehs_per_ring = num_vehicles/self.num_rings

        # if not all lanes are equal, then we must ensure that vehicles are in
        # two edges at the same time
        flag = False
        lanes = [self.num_lanes(edge) for edge in self.get_edge_list()]
        if any(lanes[0] != lanes[i] for i in range(1, len(lanes))):
            flag = True

        x = x0
        car_count = 0
        startpositions, startlanes = [], []

        # generate uniform starting positions
        while car_count < num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x)

            # place vehicles side-by-side in all available lanes on this edge
            for lane in range(min([self.num_lanes(pos[0]), lanes_distr])):
                car_count += 1
                startpositions.append(pos)
                edge, pos = startpositions[-1]
                startpositions[-1] = edge, pos % self.length
                startlanes.append(lane)

                if car_count == num_vehicles:
                    break

            # 1e-13 prevents an extra car from being added in wrong place
            x = (x + increment + VEHICLE_LENGTH + min_gap) + 1e-13
            if (car_count % vehs_per_ring) == 0:
                # if we have put in the right number of cars,
                # move onto the next ring
                ring_num = int(car_count/vehs_per_ring)
                x = self.length * ring_num + 1e-13

        # add a perturbation to each vehicle, while not letting the vehicle
        # leave its current edge
        # if initial_config.perturbation > 0:
        #     for i in range(num_vehicles):
        #         perturb = np.random.normal(0, initial_config.perturbation)
        #         edge, pos = startpositions[i]
        #         pos = max(0, min(self.edge_length(edge), pos + perturb))
        #         startpositions[i] = (edge, pos)

        return startpositions, startlanes