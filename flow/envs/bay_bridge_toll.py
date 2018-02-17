from flow.envs.lane_changing import LaneChangeAccelEnv

from gym.spaces.box import Box

MAX_LANES = 16  # largest number of lanes in the network


class BridgeTollBaseEnv(LaneChangeAccelEnv):
    def __init__(self, env_params, sumo_params, scenario):
        super().__init__(env_params, sumo_params, scenario)


class BridgeTollEnv(BridgeTollBaseEnv):
    @property
    def observation_space(self):
        num_edges = len(self.scenario.get_edge_list())
        num_rl_veh = self.vehicles.num_rl_vehicles
        num_obs = 2*num_edges + 4*MAX_LANES + 3*num_rl_veh
        return Box(low=-float("inf"), high=float("inf"), shape=(num_obs,))

    def get_state(self):
        # rl vehicles are sorted by their position
        rl_ids = sorted(self.vehicles.get_rl_ids(), key=self.get_x_by_id)

        # rl vehicle data (absolute position, speed, and lane index)
        rl_obs = []
        for veh_id in rl_ids:
            rl_obs += [self.get_x_by_id(veh_id),
                       self.vehicles.get_speed(veh_id),
                       self.vehicles.get_lane(veh_id)]

        # relative vehicles data (lane headways, tailways, vel_ahead, and
        # vel_behind)
        relative_obs = []
        for veh_id in rl_ids:
            headway = [1000 for _ in range(MAX_LANES)]
            tailway = [1000 for _ in range(MAX_LANES)]
            vel_in_front = [0 for _ in range(MAX_LANES)]
            vel_behind = [0 for _ in range(MAX_LANES)]

            lane_leaders = self.vehicles.get_lane_leaders(veh_id)
            lane_followers = self.vehicles.get_lane_followers(veh_id)
            lane_headways = self.vehicles.get_lane_headways(veh_id)
            lane_tailways = self.vehicles.get_lane_tailways(veh_id)
            headway[0:len(lane_headways)] = lane_headways
            tailway[0:len(lane_tailways)] = lane_tailways
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = self.vehicles.get_speed(lane_leader)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = self.vehicles.get_speed(lane_follower)

        # per edge data (average speed, density
        edge_obs = []
        for edge in self.scenario.get_edge_list():
            veh_ids = self.vehicles.get_ids_by_edge(edge)
            avg_speed = sum(self.vehicles.get_speed(veh_ids)) / len(veh_ids)
            density = len(veh_ids) / self.scenario.edge_length(edge)
            edge_obs += [avg_speed, density]

        return rl_obs + relative_obs + edge_obs

    def sort_by_position(self):
        if self.env_params.sort_vehicles:
            sorted_ids = sorted(self.vehicles.get_ids(),
                                key=self.get_x_by_id)
            return sorted_ids, None
        else:
            return self.vehicles.get_ids(), None
