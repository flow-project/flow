from flow.envs.base_env import Env
from flow.envs.lane_changing import LaneChangeAccelEnv
from flow.core import rewards
from flow.core import multi_agent_rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from collections import defaultdict
from copy import deepcopy

import numpy as np

EDGE_LIST = ["1", "2", "3", "4", "5"]
EDGE_BEFORE_TOLL = "1"
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"
NUM_TOLL_LANES = 8
TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # controls how close we have to be for the red light to start going off

EDGE_BEFORE_RAMP_METER = "2"
EDGE_AFTER_RAMP_METER = "3"
NUM_RAMP_METERS = 8
RAMP_METER_AREA = 80

MAX_LANES = 8  # largest number of lanes in the network

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15
FAST_TRACK_ON = range(6, 11)


class BottleneckEnv(LaneChangeAccelEnv):
    def __init__(self, env_params, sumo_params, scenario):
        self.num_rl = deepcopy(scenario.vehicles.num_rl_vehicles)
        super().__init__(env_params, sumo_params, scenario)
        self.edge_dict = defaultdict(list)
        self.cars_waiting_for_toll = dict()
        self.cars_waiting_before_ramp_meter = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step, 4 / self.sim_step, NUM_TOLL_LANES))
        self.tl_state = ""
        self.disable_tb = False
        self.disable_ramp_metering = False

        print(env_params.additional_params)
        if "disable_tb" in env_params.additional_params:
            self.disable_tb = env_params.get_additional_param("disable_tb")

        if "disable_ramp_metering" in env_params.additional_params:
            self.disable_ramp_metering = env_params.get_additional_param("disable_ramp_metering")

        print(self.disable_tb)

    def additional_command(self):
        super().additional_command()
        # build a list of vehicles and their edges and positions
        self.edge_dict = defaultdict(list)
        # update the dict with all the edges in edge_list so we can look forward for edges
        self.edge_dict.update((k, [[] for _ in range(MAX_LANES)]) for k in EDGE_LIST)
        for veh_id in self.vehicles.get_ids():
            edge = self.vehicles.get_edge(veh_id)
            if edge not in self.edge_dict:
                self.edge_dict.update({edge: [[] for _ in range(MAX_LANES)]})
            lane = self.vehicles.get_lane(veh_id)  # integer
            pos = self.vehicles.get_position(veh_id)
            self.edge_dict[edge][lane].append((veh_id, pos))
        if not self.disable_tb:
            self.apply_toll_bridge_control()
        if not self.disable_ramp_metering:
            self.ramp_meter_lane_change_control()

    def ramp_meter_lane_change_control(self):
        cars_that_have_left = []
        for veh_id in self.cars_waiting_before_ramp_meter:
            if self.vehicles.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                lane_change_mode = self.cars_waiting_before_ramp_meter[veh_id]["lane_change_mode"]
                color = self.cars_waiting_before_ramp_meter[veh_id]["color"]
                self.traci_connection.vehicle.setColor(veh_id, color)
                self.traci_connection.vehicle.setLaneChangeMode(veh_id, lane_change_mode)

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            self.cars_waiting_before_ramp_meter.__delitem__(veh_id)

        for lane in range(NUM_RAMP_METERS):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for car in cars_in_lane:
                veh_id, pos = car
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        lane_change_mode = self.vehicles.get_lane_change_mode(veh_id)
                        color = self.traci_connection.vehicle.getColor(veh_id)
                        self.cars_waiting_before_ramp_meter[veh_id] = {"lane_change_mode": lane_change_mode, "color": color}
                        self.traci_connection.vehicle.setLaneChangeMode(veh_id, 512)
                        self.traci_connection.vehicle.setColor(veh_id, (0, 255, 255, 0))

    def apply_toll_bridge_control(self):
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.vehicles.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.vehicles.get_lane(veh_id)
                lane_change_mode = self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.traci_connection.vehicle.setColor(veh_id, color)
                self.traci_connection.vehicle.setLaneChangeMode(veh_id, lane_change_mode)
                if lane not in FAST_TRACK_ON:
                    self.toll_wait_time[lane] = max(0,
                                                    np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                                                                     1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(0,
                                                    np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK / self.sim_step,
                                                                     1 / self.sim_step))

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            self.cars_waiting_for_toll.__delitem__(veh_id)

        traffic_light_states = ["G"] * NUM_TOLL_LANES

        for lane in range(NUM_TOLL_LANES):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_TOLL][lane]

            for car in cars_in_lane:
                veh_id, pos = car
                if pos > TOLL_BOOTH_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        lane_change_mode = self.vehicles.get_lane_change_mode(veh_id)
                        color = self.traci_connection.vehicle.getColor(veh_id)
                        self.cars_waiting_for_toll[veh_id] = {"lane_change_mode": lane_change_mode, "color": color}
                        self.traci_connection.vehicle.setLaneChangeMode(veh_id, 512)
                        self.traci_connection.vehicle.setColor(veh_id, (255, 0, 255, 0))
                    else:
                        if pos > 50:
                            if self.toll_wait_time[lane] < 0:
                                traffic_light_states[lane] = "G"
                            else:
                                traffic_light_states[lane] = "r"
                                self.toll_wait_time[lane] -= 1

        newTLState = "".join(traffic_light_states)

        if newTLState != self.tl_state:
            self.tl_state = newTLState
            self.traci_connection.trafficlights.setRedYellowGreenState(tlsID=TB_TL_ID, state=newTLState)


class BridgeTollEnv(BottleneckEnv):
    @property
    def observation_space(self):
        num_edges = len(self.scenario.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2*num_edges + 4*MAX_LANES*num_rl_veh + 3*num_rl_veh
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
            relative_obs += headway + tailway + vel_in_front + vel_behind

        # per edge data (average speed, density
        edge_obs = []
        for edge in self.scenario.get_edge_list():
            veh_ids = self.vehicles.get_ids_by_edge(edge)
            if len(veh_ids) > 0:
                avg_speed = sum(self.vehicles.get_speed(veh_ids)) / len(veh_ids)
                density = len(veh_ids) / self.scenario.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        extra_zeros = []
        if len(rl_ids) != self.num_rl:
            diff = (self.num_rl - len(rl_ids))
            extra_zeros = [0]*(4*MAX_LANES*diff + 3*diff)
        return np.asarray(rl_obs + relative_obs + edge_obs + extra_zeros)

    def sort_by_position(self):
        if self.env_params.sort_vehicles:
            sorted_ids = sorted(self.vehicles.get_ids(),
                                key=self.get_x_by_id)
            return sorted_ids, None
        else:
            return self.vehicles.get_ids(), None

    def apply_rl_actions(self, actions):
        """
        See parent class

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands for the duration of the lane
        change and return negative rewards for actions during that lane change.
        if a lane change isn't applied, and sufficient time has passed, issue an
        acceleration like normal.
        """
        num_rl = self.vehicles.num_rl_vehicles
        acceleration = actions[::2][:num_rl]
        direction = np.round(actions[1::2])[:num_rl]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <= self.lane_change_duration
             + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)
