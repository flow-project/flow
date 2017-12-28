from flow.envs.base_env import SumoEnvironment

import numpy as np


class IntersectionEnvironment(SumoEnvironment):
    """
    A class that may be used to design environments with intersections. Allows
    the user to calculate the distance a vehicle is from the nearest
    intersection, assuming the function "specify_intersection_edge_starts" in
    the scenario class is properly prepared.
    """

    def __init__(self, env_params, sumo_params, scenario):
        super(IntersectionEnvironment, self).__init__(
            env_params, sumo_params, scenario)

        self.intersection_edges = []
        if hasattr(self.scenario, "intersection_edgestarts"):
            for intersection_tuple in self.scenario.intersection_edgestarts:
                self.intersection_edges.append(intersection_tuple[0])

    def get_distance_to_intersection(self, veh_ids):
        """
        Determines the smallest distance from the current vehicle's position to
        any of the intersections.

        Parameters
        ----------
        veh_ids: str
            vehicle identifier

        Yields
        ------
        tup
            1st element: distance to closest intersection
            2nd element: intersection ID (which also specifies which side of the
            intersection the vehicle will be arriving at)
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
            dist.append((intersection_tuple[1] - this_pos))
            intersection.append(intersection_tuple[0])

        ind = np.argmin(np.abs(dist))

        return dist[ind], intersection[ind]

    def sort_by_intersection_dist(self):
        """
        Sorts the vehicle ids of vehicles in the network by their distance to
        the intersection.
        The base environment does this by sorting vehicles by their distance to
        intersection, as specified by the "get_distance_to_intersection"
        function.

        Returns
        -------
        sorted_ids: list
            a list of all vehicle IDs sorted by position
        sorted_extra_data: list or tuple
            an extra component (list, tuple, etc...) containing extra sorted
            data, such as positions. If no extra component is needed, a value
            of None should be returned
        """
        ids = self.vehicles.get_ids()
        sorted_indx = np.argsort(self.get_distance_to_intersection(ids))
        sorted_ids = np.array(ids)[sorted_indx]
        return sorted_ids
