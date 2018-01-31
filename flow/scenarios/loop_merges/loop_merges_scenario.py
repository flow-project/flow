import numpy as np
from numpy import pi

from flow.scenarios.base_scenario import Scenario
from flow.core.traffic_lights import TrafficLights
from flow.core.params import InitialConfig


class LoopMergesScenario(Scenario):

    def __init__(self, name, generator_class, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """
        Initializes a two-way intersection scenario.

        See scenario.py for description of params.
        """
        self.merge_in_len = net_params.additional_params["merge_in_length"]
        self.merge_out_len = net_params.additional_params["merge_out_length"]
        self.merge_in_angle = net_params.additional_params["merge_in_angle"]
        self.merge_out_angle = net_params.additional_params["merge_out_angle"]
        self.radius = net_params.additional_params["ring_radius"]

        # the vehicles that start in the merging lane are distinguished by the
        # presence of the string "merge" in their names
        self.num_merge_vehicles = \
            sum(["merge" in vehicles.get_state(veh_id, "type")
                 for veh_id in vehicles.get_ids()])

        # TODO: find a good way of calculating these
        self.ring_0_0_len = 1.1 + 4 * net_params.additional_params["lanes"]
        self.ring_1_0_len = 1.1 + 4 * net_params.additional_params["lanes"]
        self.ring_0_n_len = 6.5
        self.ring_1_n_len = 6.5
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = \
            2 * pi * self.radius + self.ring_0_n_len + self.ring_1_n_len

        if "length" not in net_params.additional_params:
            raise ValueError("length of circle not supplied")
        self.length = net_params.additional_params["length"]

        if "lanes" not in net_params.additional_params:
            raise ValueError("lanes of circle not supplied")
        self.lanes = net_params.additional_params["lanes"]

        if "speed_limit" not in net_params.additional_params:
            raise ValueError("speed limit of circle not supplied")

        if "resolution" not in net_params.additional_params:
            raise ValueError("resolution of circle not supplied")

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """
        See parent class
        """
        if self.merge_out_len is not None:
            ring_0_len = (self.merge_out_angle - self.merge_in_angle) % \
                         (2 * pi) * self.radius

            edgestarts = \
                [("ring_0",
                  self.ring_0_n_len),
                 ("ring_1",
                  self.ring_0_n_len + ring_0_len + self.ring_1_n_len),
                 ("merge_in",
                  - self.merge_in_len - self.ring_0_0_len + self.ring_0_n_len),
                 ("merge_out",
                  1000 * (2 * pi * self.radius) + self.ring_1_0_len)]

        else:
            edgestarts = \
                [("ring_0",
                  self.ring_0_n_len),
                 ("ring_1",
                  self.ring_0_n_len + pi * self.radius + self.inner_space_len),
                 ("merge_in",
                  - self.merge_in_len - self.ring_0_0_len + self.ring_0_n_len)]

        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        lanes = self.net_params.additional_params["lanes"]

        if self.merge_out_len is not None:
            ring_0_len = (self.merge_out_angle - self.merge_in_angle) % \
                         (2 * pi) * self.radius

            internal_edgestarts = \
                [(":ring_0_%d" % lanes, 0),
                 (":ring_1_%d" % lanes, self.ring_0_n_len + ring_0_len),
                 (":ring_0_0", - self.ring_0_0_len + self.ring_0_n_len),
                 (":ring_1_0", 1000 * (2 * pi * self.radius))]

        else:
            internal_edgestarts = \
                [(":ring_0_%d" % lanes, 0),
                 (":ring_1_0", self.ring_0_n_len + pi * self.radius),
                 (":ring_0_0", - self.ring_0_0_len + self.ring_0_n_len)]

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, **kwargs):
        """
        See base class
        """
        x0 = 1
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = initial_config.bunching

        lanes_distribution = initial_config.lanes_distribution

        merge_bunching = 0
        if "merge_bunching" in initial_config.additional_params:
            merge_bunching = initial_config.additional_params["merge_bunching"]

        n_merge_platoons = None
        if "n_merge_platoons" in initial_config.additional_params:
            n_merge_platoons = \
                initial_config.additional_params["n_merge_platoons"]

        startpositions = []
        startlanes = []

        # generate starting positions for non-merging vehicles
        if self.vehicles.num_vehicles - self.num_merge_vehicles > 0:
            # in order to avoid placing cars in the internal edges, their
            # length is removed from the distribution length
            distribution_len = \
                self.length - self.ring_0_n_len - self.ring_1_n_len
            increment = (distribution_len - bunching) * lanes_distribution / \
                        (self.vehicles.num_vehicles - self.num_merge_vehicles)

            x = [x0] * lanes_distribution
            car_count = 0
            lane_count = 0
            while car_count < self.vehicles.num_vehicles \
                    - self.num_merge_vehicles:
                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                if ":ring_0" in pos[0]:
                    x[lane_count] += self.ring_0_n_len
                    pos = self.get_edge(x[lane_count])
                elif ":ring_1" in pos[0]:
                    x[lane_count] += self.ring_1_n_len
                    pos = self.get_edge(x[lane_count])

                startpositions.append(pos)
                startlanes.append(lane_count)

                x[lane_count] = (x[lane_count] + increment) % self.length

                # increment the car_count and lane_count
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                if lane_count >= lanes_distribution:
                    lane_count = 0

        # generate starting positions for merging vehicles
        if self.num_merge_vehicles > 0:
            x = [self.get_x(edge="merge_in", position=0)] * lanes_distribution
            car_count = 0
            lane_count = 0
            while car_count < self.num_merge_vehicles:
                if n_merge_platoons is None:
                    # if no platooning is requested for merging vehicles, the
                    # vehicles are uniformly distributed across the appropriate
                    # section of the merge_in length
                    increment = (self.merge_in_len - merge_bunching) * \
                                lanes_distribution / self.num_merge_vehicles
                else:
                    if True:  # FIXME
                        # some small value (to ensure vehicles are bunched together)
                        increment = 8
                    else:
                        increment = 1  # FIXME

                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                startpositions.append(pos)
                startlanes.append(lane_count)

                x[lane_count] += increment

                # increment the car_count and lane_count
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles should
                # be distributed on in the network, reset
                if lane_count >= lanes_distribution:
                    lane_count = 0

        return startpositions, startlanes

    def gen_gaussian_start_pos(self, initial_config, **kwargs):
        """
        Generates start positions that are perturbed from a uniformly spaced
        distribution by some gaussian noise.
        """
        x0 = 1
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = 0
        if "bunching" in initial_config:
            bunching = initial_config["bunching"]

        lanes_distribution = 1
        if "lanes_distribution" in initial_config:
            lanes_distribution = initial_config["lanes_distribution"]

        distribution_length = self.length
        if "distribution_length" in initial_config:
            distribution_length = initial_config["distribution_length"]

        startpositions = []
        startlanes = []
        increment = (distribution_length - bunching) * lanes_distribution \
            / self.vehicles.num_vehicles

        x = [x0] * lanes_distribution
        x_start = np.array([])
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles:
            # collect the position and lane number of each new vehicle
            x_start = np.append(x_start, x[lane_count])
            startlanes.append(lane_count)

            x[lane_count] = (x[lane_count] + increment) % distribution_length

            # increment the car_count and lane_num
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        # add noise to uniform starting positions
        for i in range(len(x_start)):
            # perturbation from uniform distribution
            x_start[i] =\
                (x_start[i] +
                 min(2.5, max(-2.5, np.random.normal(loc=0, scale=2.5)))) \
                % distribution_length

            pos = self.get_edge(x_start[i])
            startpositions.append(pos)

        return startpositions, startlanes

    def gen_gaussian_additive_start_pos(self, initial_config, **kwargs):
        """
        Generate random start positions via additive Gaussian.

        WARNING: this does not absolutely gaurantee that the order of
        vehicles is preserved.

        Returns
        -------
        startpositions : list
            start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list
            start lanes
        """
        x0 = 1
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        bunching = 0
        if "bunching" in initial_config:
            bunching = kwargs["bunching"]

        lanes_distribution = 1
        if "lanes_distribution" in initial_config:
            lanes_distribution = kwargs["lanes_distribution"]

        downscale = 5
        if "downscale" in initial_config:
            downscale = kwargs["downscale"]

        merge_bunching = 0
        if "merge_bunching" in initial_config:
            merge_bunching = kwargs["merge_bunching"]

        startpositions = []
        startlanes = []

        # generate starting positions for non-merging vehicles
        # in order to avoid placing cars in the internal edges, their length is
        # removed from the distribution length
        distribution_len = self.length - self.ring_0_n_len - self.ring_1_n_len
        mean = (distribution_len - bunching) * lanes_distribution / \
            (self.vehicles.num_vehicles - self.num_merge_vehicles)

        x = [x0] * lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.vehicles.num_vehicles - self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            if ":ring_0" in pos[0]:
                x[lane_count] += self.ring_0_n_len
                pos = self.get_edge(x[lane_count])
            elif ":ring_1" in pos[0]:
                x[lane_count] += self.ring_1_n_len
                pos = self.get_edge(x[lane_count])

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] = \
                (x[lane_count] +
                 np.random.normal(scale=mean/downscale, loc=mean)) % self.length

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        # generate starting positions for merging vehicles
        mean = (self.merge_in_len - merge_bunching) * lanes_distribution \
            / self.num_merge_vehicles

        x = [self.get_x(edge="merge_in", position=0)] * lanes_distribution
        car_count = 0
        lane_count = 0
        while car_count < self.num_merge_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x[lane_count])

            startpositions.append(pos)
            startlanes.append(lane_count)

            x[lane_count] += np.random.normal(scale=mean / downscale, loc=mean)

            # increment the car_count and lane_count
            car_count += 1
            lane_count += 1
            # if the lane num exceeds the number of lanes the vehicles should
            # be distributed on in the network, reset
            if lane_count >= lanes_distribution:
                lane_count = 0

        return startpositions, startlanes
