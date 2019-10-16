from flow.envs import TestEnv

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3369, 3341, 3370, 3344, 3329]}


class CoordinatedEnv(TestEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]
        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        for node_id in self.target_nodes:
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)
            for edge_id in incoming_edges:
                stopbar_ids, advanced_ids = self.k.traffic_light.get_detectors_on_edge(edge_id)
                type_map = {"stopbar_ids": stopbar_ids, "advanced_ids": advanced_ids}

                self.edge_detector_dict[node_id] = {edge_id: type_map}
            # self.edges_with_detectors.update(incoming_edges)

    def additional_command(self):
        """Additional commands that may be performed by the step method."""
        # print(self.edge_detector_dict)
        veh_types = ["Car"]
        self.k.vehicle.tracked_vehicle_types.update(veh_types)
        # tl_ids = self.k.traffic_light.get_ids()
        # print(tl_ids)
        print(self.k.traffic_light.set_intersection_offset(3344, -20))
