from flow.scenarios.base_scenario import Scenario

from numpy import pi, arcsin


class TwoLoopsOneMergingScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=None):
        """
        Initializes a two loop scenario where one loop merging in and out of
        the other. Required net_params: ring_radius, lanes, speed_limit,
        resolution.

        See Scenario.py for description of params.
        """
        radius = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        self.junction_length = 0.3
        self.intersection_length = 25.5  # calibrate when the radius changes

        net_params.additional_params["length"] = \
            2 * x + 2 * pi * radius + \
            2 * self.intersection_length + 2 * self.junction_length

        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config)

    def specify_edge_starts(self):
        """
        See parent class
        """
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        edgestarts = [
            ("left", self.intersection_length),
            ("center", ring_edgelen + 2 * self.intersection_length),
            ("bottom", 2 * ring_edgelen + 2 * self.intersection_length),
            ("right", 2 * ring_edgelen + lane_length + 2 * self.intersection_length + self.junction_length),
            ("top", 3 * ring_edgelen + lane_length + 2 * self.intersection_length + 2 * self.junction_length)
        ]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        r = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        ring_edgelen = pi * r

        # internal_edgestarts = [(":", -1)]
        internal_edgestarts = [
            (":top_left", 0),
            (":bottom_left", ring_edgelen + self.intersection_length),
            (":bottom_right", 2 * ring_edgelen + lane_length + 2 * self.intersection_length),
            (":top_right", 3 * ring_edgelen + lane_length + 2 * self.intersection_length + self.junction_length)
        ]

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, **kwargs):
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

        print(x0)

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
        self.n_inner_vehicles = num_merge_vehicles
        self.n_outer_vehicles = num_vehicles - num_merge_vehicles

        radius = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        startpositions = []
        startlanes = []
        length_loop = 2 * pi * radius
        self.length_loop = length_loop

        try:
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
                    # find the location of the internal edge in
                    # total_edgestarts, which has the edges ordered by position
                    edges = [tup[0] for tup in self.total_edgestarts]
                    indx_edge = \
                        [i for i in range(len(edges)) if edges[i] == pos[0]][0]

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
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                if lane_count >= initial_config.lanes_distribution:
                    lane_count = 0
        except ZeroDivisionError:
            pass

        length_merge = pi * radius + 2 * lane_length
        try:
            increment_merge = \
                (length_merge - merge_bunching) * \
                initial_config.lanes_distribution / num_merge_vehicles

            x = [dict(self.edgestarts)["bottom"]] * \
                initial_config.lanes_distribution
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
                    indx_edge = [i for i in range(len(edges)) if edges[i] == pos[0]][0]

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
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                if lane_count >= initial_config.lanes_distribution:
                    lane_count = 0
        except ZeroDivisionError:
            pass

        return startpositions, startlanes
