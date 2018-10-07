"""Contains the loop merge scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

from numpy import pi
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # radius of the loops
    "ring_radius": 50,
    # length of the straight edges connected the outer loop to the inner loop
    "lane_length": 75,
    # number of lanes in the inner loop
    "inner_lanes": 3,
    # number of lanes in the outer loop
    "outer_lanes": 2,
    # max speed limit in the network
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40,
}


class TwoLoopsOneMergingScenario(Scenario):
    """Two loop merge scenario."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a two loop scenario.

        Requires from net_params:
        - ring_radius: radius of the loops
        - lane_length: length of the straight edges connected the outer loop to
          the inner loop
        - inner_lanes: number of lanes in the inner loop
        - outer_lanes: number of lanes in the outer loop
        - speed_limit: max speed limit in the network
        - resolution: resolution of the curved portions

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        radius = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        self.junction_length = 0.3
        self.intersection_length = 25.5  # calibrate when the radius changes

        net_params.additional_params["length"] = \
            2 * x + 2 * pi * radius + \
            2 * self.intersection_length + 2 * self.junction_length

        num_vehicles = vehicles.num_vehicles
        num_merge_vehicles = sum("merge" in vehicles.get_state(veh_id, "type")
                                 for veh_id in vehicles.get_ids())
        self.n_inner_vehicles = num_merge_vehicles
        self.n_outer_vehicles = num_vehicles - num_merge_vehicles

        radius = net_params.additional_params["ring_radius"]
        length_loop = 2 * pi * radius
        self.length_loop = length_loop

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        edgestarts = [
            ("left", self.intersection_length),
            ("center", ring_edgelen + 2 * self.intersection_length),
            ("bottom", 2 * ring_edgelen + 2 * self.intersection_length),
            ("right", 2 * ring_edgelen + lane_length +
             2 * self.intersection_length + self.junction_length),
            ("top", 3 * ring_edgelen + lane_length +
             2 * self.intersection_length + 2 * self.junction_length)
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        """See parent class."""
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        internal_edgestarts = [
            (":top_left", 0), (":bottom_left",
                               ring_edgelen + self.intersection_length),
            (":bottom_right",
             2 * ring_edgelen + lane_length + 2 * self.intersection_length),
            (":top_right", 3 * ring_edgelen + lane_length +
             2 * self.intersection_length + self.junction_length)
        ]

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """See parent class.

        Vehicles with the prefix "merge" are placed in the merge ring,
        while all other vehicles are placed in the ring.
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        random_scale = \
            self.initial_config.additional_params.get("gaussian_scale", 0)

        bunching = initial_config.bunching
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        merge_bunching = 0
        if "merge_bunching" in initial_config.additional_params:
            merge_bunching = initial_config.additional_params["merge_bunching"]

        num_vehicles = self.vehicles.num_vehicles
        num_merge_vehicles = \
            sum("merge" in self.vehicles.get_state(veh_id, "type")
                for veh_id in self.vehicles.get_ids())

        radius = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        startpositions = []
        startlanes = []
        length_loop = 2 * pi * radius

        try:
            increment_loop = \
                (self.length_loop - bunching) \
                * self.net_params.additional_params["inner_lanes"] \
                / (num_vehicles - num_merge_vehicles)

            # x = [x0] * initial_config.lanes_distribution
            if self.initial_config.additional_params.get(
                    "ring_from_right", False):
                x = [dict(self.edgestarts)["right"]] * \
                    self.net_params.additional_params["inner_lanes"]
            else:
                x = [x0] * self.net_params.additional_params["inner_lanes"]
            car_count = 0
            lane_count = 0
            while car_count < num_vehicles - num_merge_vehicles:
                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                # ensures that vehicles are not placed in an internal junction
                while pos[0] in dict(self.internal_edgestarts).keys():
                    # find the location of the internal edge in
                    # total_edgestarts, which has the edges ordered by position
                    edges = [tup[0] for tup in self.total_edgestarts]
                    indx_edge = next(
                        i for i, edge in enumerate(edges) if edge == pos[0])

                    # take the next edge in the list, and place the car at the
                    # beginning of this edge
                    if indx_edge == len(edges) - 1:
                        next_edge_pos = self.total_edgestarts[0]
                    else:
                        next_edge_pos = self.total_edgestarts[indx_edge + 1]

                    x[lane_count] = next_edge_pos[1]
                    pos = (next_edge_pos[0], 0)

                startpositions.append(pos)
                startlanes.append(lane_count)

                x[lane_count] = \
                    (x[lane_count] + increment_loop
                     + random_scale * np.random.randn()) % length_loop

                # increment the car_count and lane_num
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                if lane_count >= \
                        self.net_params.additional_params["inner_lanes"]:
                    lane_count = 0
        except ZeroDivisionError:
            pass

        length_merge = pi * radius + 2 * lane_length
        try:
            increment_merge = \
                (length_merge - merge_bunching) * \
                initial_config.lanes_distribution / num_merge_vehicles

            if self.initial_config.additional_params.get(
                    "merge_from_top", False):
                x = [dict(self.edgestarts)["top"] - x0] * \
                    self.net_params.additional_params["outer_lanes"]
            else:
                x = [dict(self.edgestarts)["bottom"] - x0] * \
                    self.net_params.additional_params["outer_lanes"]
            car_count = 0
            lane_count = 0
            while car_count < num_merge_vehicles:
                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                # ensures that vehicles are not placed in an internal junction
                while pos[0] in dict(self.internal_edgestarts).keys():
                    # find the location of the internal edge in
                    # total_edgestarts, which has the edges ordered by position
                    edges = [tup[0] for tup in self.total_edgestarts]
                    indx_edge = next(
                        i for i, edge in enumerate(edges) if edge == pos[0])

                    # take the next edge in the list, and place the car at the
                    # beginning of this edge
                    if indx_edge == len(edges) - 1:
                        next_edge_pos = self.total_edgestarts[0]
                    else:
                        next_edge_pos = self.total_edgestarts[indx_edge + 1]

                    x[lane_count] = next_edge_pos[1]
                    pos = (next_edge_pos[0], 0)

                startpositions.append(pos)
                startlanes.append(lane_count)

                if self.initial_config.additional_params.get(
                        "merge_from_top", False):
                    x[lane_count] = x[lane_count] - increment_merge + \
                        random_scale*np.random.randn()
                else:
                    x[lane_count] = x[lane_count] + increment_merge + \
                        random_scale*np.random.randn()

                # increment the car_count and lane_num
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                # if lane_count >= self.initial_config.lane_distribution
                if lane_count >= \
                        self.net_params.additional_params["outer_lanes"]:
                    lane_count = 0

        except ZeroDivisionError:
            pass

        # all vehicles start with an initial speed of 0 m/s
        startvel = [0 for _ in range(len(startlanes))]

        return startpositions, startlanes, startvel
