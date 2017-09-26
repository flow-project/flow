from cistar.envs.base_env import SumoEnvironment

import numpy as np

class IntersectionEnvironment(SumoEnvironment):

    def __init__(self, env_params, sumo_params, scenario):
        super(IntersectionEnvironment, self).__init__(env_params, sumo_params, scenario)

        self.intersection_edges = []
        if hasattr(self.scenario, "intersection_edgestarts"):
            for intersection_tuple in self.scenario.intersection_edgestarts:
                self.intersection_edges.append(intersection_tuple[0])

    def get_distance_to_intersection(self, veh_ids):
        """
        Determines the smallest distance from the current vehicle's position to any of the intersections.
        :param veh_id: vehicle identifier
        :return: a tuple containing the distance to the intersection and which side of the
                 intersection the vehicle will be arriving at.
        """
        if isinstance(veh_ids, list):
            return [self.find_intersection_dist(veh_id)[0] for veh_id in veh_ids]
        else:
            return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        this_pos = self.get_x_by_id(veh_id)

        if not self.scenario.intersection_edgestarts:
            raise ValueError("The scenario does not contain intersections.")

        dist = []
        intersection = []
        for intersection_tuple in self.scenario.intersection_edgestarts:
            # dist.append((intersection_tuple[1] - this_pos) % self.scenario.length)
            dist.append((intersection_tuple[1] - this_pos))
            intersection.append(intersection_tuple[0])

        ind = np.argmin(np.abs(dist))

        return dist[ind], intersection[ind]

    def sort_by_intersection_dist(self):
        """
        Sorts the vehicle ids of vehicles in the network by their distance to the intersection.
        The base environment does this by sorting vehicles by their distance to intersection, as specified
        by the "get_distance_to_intersection" function.

        :return: a list of sorted vehicle ids
                 an extra component (list, tuple, etc...) containing extra sorted data, such as positions.
                  If no extra component is needed, a value of None should be returned
        """
        sorted_indx = np.argsort(self.get_distance_to_intersection(self.ids))
        sorted_ids = np.array(self.ids)[sorted_indx]
        return sorted_ids

