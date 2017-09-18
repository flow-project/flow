from cistar.envs.base_env import SumoEnvironment

import numpy as np


class IntersectionEnvironment(SumoEnvironment):
    """
    A class that may be used to design environments with intersections.
    """

    def __init__(self, env_params, sumo_params, scenario):
        super(IntersectionEnvironment, self).__init__(env_params, sumo_params, scenario)

        self.intersection_edges = []
        if hasattr(self.scenario, "intersection_edgestarts"):
            for intersection_tuple in self.scenario.intersection_edgestarts:
                self.intersection_edges.append(intersection_tuple[0])

    def get_distance_to_intersection(self, veh_id):
        """
        Determines the smallest distance from the current vehicle's position to any of the intersections.

        :param veh_id: vehicle identifier
        :return: a tuple containing the distance to the intersection and which side of the
                 intersection the vehicle will be arriving at.
        """
        this_pos = self.get_x_by_id(veh_id)

        if not self.scenario.intersection_edgestarts:
            raise ValueError("The scenario does not contain intersections.")

        dist = []
        intersection = []
        for intersection_tuple in self.scenario.intersection_edgestarts:
            dist.append((intersection_tuple[1] - this_pos) % self.scenario.length)
            intersection.append(intersection_tuple[0])

        ind = np.argmin(dist)

        return dist[ind], intersection[ind]
