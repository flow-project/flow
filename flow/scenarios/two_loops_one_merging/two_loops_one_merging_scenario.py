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
        radius_merge = 1.5 * radius
        angle_merge = arcsin(0.75)
        net_params.additional_params["length"] = \
            2 * pi * radius + 2 * (pi - angle_merge) * radius

        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config)

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
            if ":left" in pos[0]:
                x[lane_count] += 0.3
                pos = self.get_edge(x[lane_count])

            elif ":bottom" in pos[0]:
                x[lane_count] += 7
                pos = self.get_edge(x[lane_count])

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
            if ":right" in pos[0]:
                x[lane_count] += 0.3
                pos = self.get_edge(x[lane_count])

            elif ":top" in pos[0]:
                x[lane_count] += 7
                pos = self.get_edge(x[lane_count])

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
