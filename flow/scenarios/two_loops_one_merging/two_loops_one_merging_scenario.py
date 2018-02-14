from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

from numpy import pi, arcsin


ADDITIONAL_NET_PARAMS = {
    # radius of the smaller ring road (the larger has 1.5x this radius)
    "ring_radius": 230 / (2*pi),
    # number of lanes in the network
    "lanes": 1,
    # max speed limit in the network
    "speed_limit": 30,
    # number of nodes resolution
    "resolution": 0
}


class TwoLoopsOneMergingScenario(Scenario):
    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initializes a two loop scenario where one loop merging in and out of
        the other.

        Requires from net_params:
        - ring_radius: radius of the smaller ring road (the larger has 1.5x this
          radius)
        - lanes: number of lanes in the network
        - speed_limit: max speed limit in the network
        - resolution: number of nodes resolution

        See Scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        radius = net_params.additional_params["ring_radius"]
        radius_merge = 1.5 * radius
        angle_merge = arcsin(0.75)
        net_params.additional_params["length"] = \
            2 * pi * radius + 2 * (pi - angle_merge) * radius_merge

        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """
        See parent class
        """
        r = self.net_params.additional_params["ring_radius"]

        angle_small = pi/3
        ring_edgelen = (pi-angle_small) * r

        angle_large = arcsin(0.75)
        merge_edgelen = (pi - angle_large) * (1.5 * r)

        edgestarts = [("left_top", 0),
                      ("left_bottom", ring_edgelen + 0.3),
                      ("center", 2 * ring_edgelen + 7.3),
                      ("right_bottom", 3 * ring_edgelen + 14.3),
                      ("right_top", 3 * ring_edgelen + merge_edgelen + 14.6)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        r = self.net_params.additional_params["ring_radius"]

        angle_small = pi/3
        ring_edgelen = (pi-angle_small) * r

        angle_large = arcsin(0.75)
        merge_edgelen = (pi - angle_large) * (1.5 * r)

        internal_edgestarts = [
            (":left", ring_edgelen),
            (":bottom", 2 * ring_edgelen + 0.3),
            (":top", 3 * ring_edgelen + 7.3),
            (":right", 3 * ring_edgelen + merge_edgelen + 14.3)]

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """
        See parent class

        Vehicles with the prefix "merge" are placed in the merge ring,
        while all other vehicles are placed in the ring.
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

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
            sum(["merge" in self.vehicles.get_state(veh_id, "type")
                 for veh_id in self.vehicles.get_ids()])

        radius = self.net_params.additional_params["ring_radius"]
        angle_large = arcsin(0.75)

        startpositions = []
        startlanes = []
        length_loop = 2 * pi * radius
        increment_loop = \
            (length_loop - bunching) * initial_config.lanes_distribution / \
            (num_vehicles - num_merge_vehicles)

        x = [x0] * initial_config.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < num_vehicles - num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in an internal junction
            while pos[0] in dict(self.internal_edgestarts).keys():
                # find the location of the internal edge in total_edgestarts,
                # which has the edges ordered by position
                edges = [tup[0] for tup in self.total_edgestarts]
                indx_edge = next(i for i, edge in enumerate(edges)
                                 if edge == pos[0])

                # take the next edge in the list, and place the car at the
                # beginning of this edge
                if indx_edge == len(edges)-1:
                    next_edge_pos = self.total_edgestarts[0]
                else:
                    next_edge_pos = self.total_edgestarts[indx_edge+1]

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment_loop) % length_loop

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be
            # distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        length_merge = 2 * (pi - angle_large) * (1.5 * radius)
        increment_merge = \
            (length_merge - merge_bunching) * \
            initial_config.lanes_distribution / num_merge_vehicles

        x = [dict(self.edgestarts)["right_bottom"]] * \
            initial_config.lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            # ensures that vehicles are not placed in an internal junction
            while pos[0] in dict(self.internal_edgestarts).keys():
                # find the location of the internal edge in total_edgestarts,
                # which has the edges ordered by position
                edges = [tup[0] for tup in self.total_edgestarts]
                indx_edge = next(i for i, edge in enumerate(edges)
                                 if edge == pos[0])

                # take the next edge in the list, and place the car at the
                # beginning of this edge
                if indx_edge == len(edges)-1:
                    next_edge_pos = self.total_edgestarts[0]
                else:
                    next_edge_pos = self.total_edgestarts[indx_edge+1]

                x[lane_count] = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = x[lane_count] + increment_merge

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should be
            # distributed on in the network, reset
            if lane_count >= initial_config.lanes_distribution:
                lane_count = 0

        return startpositions, startlanes
