"""
Environments for training vehicles to reduce capacity drops in a bottleneck.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SumoLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import InFlows, NetParams
from flow.core.vehicles import Vehicles

from collections import defaultdict
from copy import deepcopy

import numpy as np
from gym.spaces.box import Box

from flow.core import rewards
from flow.envs.base_env import Env
import os
import glob

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth
EDGE_BEFORE_TOLL = "1"
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"
NUM_TOLL_LANES = MAX_LANES

TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # how close for the ramp meter to start going off

EDGE_BEFORE_RAMP_METER = "2"
EDGE_AFTER_RAMP_METER = "3"
NUM_RAMP_METERS = MAX_LANES

RAMP_METER_AREA = 80

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
}

# Keys for VSL style experiments
ADDITIONAL_VSL_ENV_PARAMS = {
    # number of controlled regions for velocity bottleneck controller
    "controlled_segments": [("1", 1, True), ("2", 1, True), ("3", 1, True),
                            ("4", 1, True), ("5", 1, True)],
    # whether lanes in a segment have the same action or not
    "symmetric":
    False,
    # which edges are observed
    "observed_segments": [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1)],
    # whether the inflow should be reset on each rollout
    "reset_inflow":
    False,
    # the range of inflows to reset on
    "inflow_range": [1000, 2000]
}

ADDITIONAL_NET_PARAMS = {
    "scaling": 1  # the factor multiplying number of lanes.
}

START_RECORD_TIME = 0.0
PERIOD = 10.0


class BottleneckEnv(Env):
    def __init__(self, env_params, sumo_params, scenario):
        """Environment used as a simplified representation of the toll booth
        portion of the bay bridge. Contains ramp meters, and a toll both.

        Additional
        ----------
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
        """
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in scenario.net_params.additional_params:
                raise KeyError('Net parameter "{}" not supplied'.format(p))

        self.num_rl = deepcopy(scenario.vehicles.num_rl_vehicles)
        super().__init__(env_params, sumo_params, scenario)
        env_add_params = self.env_params.additional_params
        # tells how scaled the number of lanes are
        self.scaling = scenario.net_params.additional_params.get("scaling")
        self.edge_dict = defaultdict(list)
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES * self.scaling))
        self.fast_track_lanes = range(
            int(np.ceil(1.5 * self.scaling)), int(np.ceil(2.6 * self.scaling)))

        self.tl_state = ""
        self.disable_tb = env_params.get_additional_param("disable_tb")
        self.disable_ramp_metering = \
            env_params.get_additional_param("disable_ramp_metering")
        self.rl_id_list = deepcopy(self.vehicles.get_rl_ids())

        self.next_period = START_RECORD_TIME / self.sim_step
        self.cars_arrived = 0

        # values for the ramp meter
        self.n_crit = env_add_params.get("n_crit", 8)
        self.q_max = env_add_params.get("q_max", 1100)
        self.q_min = env_add_params.get("q_min", .25 * 1100)
        self.q = self.q_min  # ramp meter feedback controller
        self.feedback_update_time = env_add_params.get("feedback_update", 15)
        self.feedback_timer = 0.0
        self.cycle_time = 6
        cycle_offset = 8
        self.ramp_state = np.linspace(0,
                                      cycle_offset * self.scaling * MAX_LANES,
                                      self.scaling * MAX_LANES)
        self.green_time = 4
        self.red_min = 2
        self.feedback_coeff = env_add_params.get("feedback_coeff", 20)

        self.smoothed_num = np.zeros(10)  # averaged number of vehs in '4'
        self.outflow_index = 0

    def additional_command(self):
        # print(self.vehicles.get_outflow_rate(100))
        super().additional_command()
        # build a list of vehicles and their edges and positions
        self.edge_dict = defaultdict(list)
        # update the dict with all the edges in edge_list
        # so we can look forward for edges
        self.edge_dict.update((k, [[]
                                   for _ in range(MAX_LANES * self.scaling)])
                              for k in EDGE_LIST)
        for veh_id in self.vehicles.get_ids():
            try:
                edge = self.vehicles.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict.update({
                        edge: [[] for _ in range(MAX_LANES * self.scaling)]
                    })
                lane = self.vehicles.get_lane(veh_id)  # integer
                pos = self.vehicles.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except Exception:
                pass
        if not self.disable_tb:
            self.apply_toll_bridge_control()
        if not self.disable_ramp_metering:
            self.ramp_meter_lane_change_control()
            self.alinea()

        # compute the outflow
        veh_ids = self.vehicles.get_ids_by_edge('4')
        self.smoothed_num[self.outflow_index] = len(veh_ids)
        self.outflow_index = \
            (self.outflow_index + 1) % self.smoothed_num.shape[0]

        if self.time_counter > self.next_period:
            self.density = self.cars_arrived  # / (PERIOD/self.sim_step)
            # print(self.density)
            self.next_period += PERIOD / self.sim_step
            self.cars_arrived = 0

        self.cars_arrived += self.vehicles.get_num_arrived()

    def ramp_meter_lane_change_control(self):
        cars_that_have_left = []
        for veh_id in self.cars_before_ramp:
            if self.vehicles.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                lane_change_mode = \
                    self.cars_before_ramp[veh_id]["lane_change_mode"]
                color = self.cars_before_ramp[veh_id]["color"]
                self.traci_connection.vehicle.setColor(veh_id, color)
                self.traci_connection.vehicle.setLaneChangeMode(
                    veh_id, lane_change_mode)

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            self.cars_before_ramp.__delitem__(veh_id)

        for lane in range(NUM_RAMP_METERS * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for car in cars_in_lane:
                veh_id, pos = car
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        traci_veh = self.traci_connection.vehicle
                        # Disable lane changes inside Toll Area
                        lane_change_mode = \
                            self.vehicles.get_lane_change_mode(veh_id)
                        color = traci_veh.getColor(veh_id)
                        self.cars_before_ramp[veh_id] = {
                            "lane_change_mode": lane_change_mode,
                            "color": color
                        }
                        traci_veh.setLaneChangeMode(veh_id, 512)
                        traci_veh.setColor(veh_id, (0, 255, 255, 255))

    def alinea(self):
        """Implementation of ALINEA from Toll Plaza Merging Traffic Control
           for Throughput Maximization"""
        self.feedback_timer += self.sim_step
        self.ramp_state += self.sim_step
        if self.feedback_timer > self.feedback_update_time:
            self.feedback_timer = 0
            # now implement the integral controller update
            # find all the vehicles in an edge
            q_update = self.feedback_coeff * (
                self.n_crit - np.average(self.smoothed_num))
            self.q = np.clip(
                self.q + q_update, a_min=self.q_min, a_max=self.q_max)
            # convert q to cycle time
            self.cycle_time = 7200 / self.q

        # now apply the ramp meter
        self.ramp_state %= self.cycle_time
        # step through, if the value of tl_state is below self.green_time
        # we should be green, otherwise we should be red
        tl_mask = (self.ramp_state <= self.green_time)
        colors = ['G' if val else 'r' for val in tl_mask]
        self.traffic_lights.set_state('3', ''.join(colors), self)

    def apply_toll_bridge_control(self):
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.vehicles.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.vehicles.get_lane(veh_id)
                lane_change_mode = \
                    self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.traci_connection.vehicle.setColor(veh_id, color)
                self.traci_connection.vehicle.setLaneChangeMode(
                    veh_id, lane_change_mode)
                if lane not in self.fast_track_lanes:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                            1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK /
                            self.sim_step, 1 / self.sim_step))

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            self.cars_waiting_for_toll.__delitem__(veh_id)

        traffic_light_states = ["G"] * NUM_TOLL_LANES * self.scaling

        for lane in range(NUM_TOLL_LANES * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_TOLL][lane]

            for car in cars_in_lane:
                veh_id, pos = car
                if pos > TOLL_BOOTH_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        lane_change_mode = \
                            self.vehicles.get_lane_change_mode(veh_id)
                        color = self.traci_connection.vehicle.getColor(veh_id)
                        self.cars_waiting_for_toll[veh_id] = \
                            {"lane_change_mode": lane_change_mode,
                             "color": color}
                        self.traci_connection.vehicle.setLaneChangeMode(
                            veh_id, 512)
                        self.traci_connection.vehicle.setColor(
                            veh_id, (255, 0, 255, 0))
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
            self.traci_connection.trafficlight.setRedYellowGreenState(
                tlsID=TB_TL_ID, state=newTLState)

    def distance_to_bottleneck(self, veh_id):
        pre_bottleneck_edges = {
            str(i): self.scenario.edge_length(str(i))
            for i in [1, 2, 3]
        }
        edge_pos = self.vehicles.get_position(veh_id)
        edge = self.vehicles.get_edge(veh_id)
        if edge in pre_bottleneck_edges:
            total_length = pre_bottleneck_edges[edge] - edge_pos
            for next_edge in range(int(edge) + 1, 4):
                total_length += pre_bottleneck_edges[str(next_edge)]
            return total_length
        else:
            return -1

    def get_bottleneck_outflow_vehicles_per_hour(self, sample_period):
        return self.vehicles.get_outflow_rate(sample_period)

    def get_bottleneck_density(self, lanes=None):
        BOTTLE_NECK_LEN = 280
        bottleneck_ids = self.vehicles.get_ids_by_edge(['3', '4'])
        if lanes:
            veh_ids = [
                veh_id for veh_id in bottleneck_ids
                if str(self.vehicles.get_edge(veh_id)) + "_" +
                str(self.vehicles.get_lane(veh_id)) in lanes
            ]
        else:
            veh_ids = self.vehicles.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

    def get_avg_bottleneck_velocity(self):
        veh_ids = self.vehicles.get_ids_by_edge(['3', '4', '5'])
        return sum(self.vehicles.get_speed(veh_ids)) / len(veh_ids) \
            if len(veh_ids) != 0 else 0

    # Dummy action and observation spaces
    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    def compute_reward(self, state, rl_actions, **kwargs):
        """ Outflow rate over last ten seconds normalized to max of 1 """

        reward = self.vehicles.get_outflow_rate(10 * self.sim_step) / \
            (2000.0 * self.scaling)
        return reward

    def get_state(self):
        """See class definition."""
        return np.asarray([1])


class BottleNeckAccelEnv(BottleneckEnv):
    """Environment used to train vehicles to effectively
       pass through a bottleneck.

       States
           An observation is the edge position, speed, lane, and edge number of
           the AV, the distance to and velocity of the vehicles
           in front and behind the AV for all lanes. Additionally, we pass the
           density and average velocity of all edges. Finally, we pad with
           zeros in case an AV has exited the system.
           Note: the vehicles are arranged in an initial order, so we pad
           the missing vehicle at its normal position in the order

       Actions
           The action space consist of a list in which the first half
           is accelerations and the second half is a direction for lane
           changing that we round

       Rewards
           The reward is the two-norm of the difference between the speed of
           all vehicles in the network and some desired speed. To this we add
           a positive reward for moving the vehicles forward

       Termination
           A rollout is terminated once the time horizon is reached.

       """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sumo_params, scenario)
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")

    @property
    def observation_space(self):
        """See class definition."""
        num_edges = len(self.scenario.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling \
            * num_rl_veh + 4 * num_rl_veh

        return Box(low=0, high=1, shape=(num_obs, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        headway_scale = 1000

        rl_ids = self.vehicles.get_rl_ids()

        # rl vehicle data (absolute position, speed, and lane index)
        rl_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list.index(veh_id)
            if rl_id_num != id_counter:
                rl_obs = np.concatenate(
                    (rl_obs, np.zeros(4 * (rl_id_num - id_counter))))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1

            # get the edge and convert it to a number
            edge_num = self.vehicles.get_edge(veh_id)
            if edge_num is None:
                edge_num = -1
            elif edge_num == '':
                edge_num = -1
            elif edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = int(edge_num) / 6
            rl_obs = np.concatenate((rl_obs, [
                self.get_x_by_id(veh_id) / 1000,
                (self.vehicles.get_speed(veh_id) / self.max_speed),
                (self.vehicles.get_lane(veh_id) / MAX_LANES), edge_num
            ]))
        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(rl_obs.shape[0] / 4)
        if diff > 0:
            rl_obs = np.concatenate((rl_obs, np.zeros(4 * diff)))

        # relative vehicles data (lane headways, tailways, vel_ahead, and
        # vel_behind)
        relative_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list.index(veh_id)
            if rl_id_num != id_counter:
                pad_mat = np.zeros(
                    4 * MAX_LANES * self.scaling * (rl_id_num - id_counter))
                relative_obs = np.concatenate((relative_obs, pad_mat))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1
            num_lanes = MAX_LANES * self.scaling
            headway = np.asarray([1000
                                  for _ in range(num_lanes)]) / headway_scale
            tailway = np.asarray([1000
                                  for _ in range(num_lanes)]) / headway_scale
            vel_in_front = np.asarray([0 for _ in range(num_lanes)
                                       ]) / self.max_speed
            vel_behind = np.asarray([0 for _ in range(num_lanes)
                                     ]) / self.max_speed

            lane_leaders = self.vehicles.get_lane_leaders(veh_id)
            lane_followers = self.vehicles.get_lane_followers(veh_id)
            lane_headways = self.vehicles.get_lane_headways(veh_id)
            lane_tailways = self.vehicles.get_lane_tailways(veh_id)
            headway[0:len(lane_headways)] = (
                np.asarray(lane_headways) / headway_scale)
            tailway[0:len(lane_tailways)] = (
                np.asarray(lane_tailways) / headway_scale)
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (
                        self.vehicles.get_speed(lane_leader) / self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = (self.vehicles.get_speed(lane_follower) /
                                     self.max_speed)

            relative_obs = np.concatenate((relative_obs, headway, tailway,
                                           vel_in_front, vel_behind))

        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(relative_obs.shape[0] / (4 * MAX_LANES))
        if diff > 0:
            relative_obs = np.concatenate((relative_obs,
                                           np.zeros(4 * MAX_LANES * diff)))

        # per edge data (average speed, density
        edge_obs = []
        for edge in self.scenario.get_edge_list():
            veh_ids = self.vehicles.get_ids_by_edge(edge)
            if len(veh_ids) > 0:
                avg_speed = (sum(self.vehicles.get_speed(veh_ids)) /
                             len(veh_ids)) / self.max_speed
                density = len(veh_ids) / self.scenario.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        return np.concatenate((rl_obs, relative_obs, edge_obs))

    def compute_reward(self, state, rl_actions, **kwargs):
        """See class definition."""
        num_rl = self.vehicles.num_rl_vehicles
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return (rewards.desired_velocity(self) + rewards.rl_forward_progress(
            self, gain=0.1) - rewards.boolean_action_penalty(
                lane_change_acts, gain=1.0))

    def sort_by_position(self):
        if self.env_params.sort_vehicles:
            sorted_ids = sorted(self.vehicles.get_ids(), key=self.get_x_by_id)
            return sorted_ids, None
        else:
            return self.vehicles.get_ids(), None

    def _apply_rl_actions(self, actions):
        """
        See parent class.

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands
        for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied,
        and sufficient time has passed, issue an acceleration like normal.
        """
        num_rl = self.vehicles.num_rl_vehicles
        acceleration = actions[::2][:num_rl]
        direction = np.round(actions[1::2])[:num_rl]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.vehicles.get_rl_ids()
        ]

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

    def additional_command(self):
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.vehicles.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.vehicles.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.traci_connection.vehicle.addFull(
                        rl_id,
                        'route1',
                        typeID=str('rl'),
                        departLane=str(lane_num),
                        departPos="0",
                        departSpeed="max")
                except Exception:
                    pass


class DesiredVelocityEnv(BottleneckEnv):
    """Environment used to train vehicles to effectively pass
       through a bottleneck by specifying the velocity that RL vehicles
       should attempt to travel in certain regions of space

       States
           An observation is the number of vehicles in each lane in each
           segment

       Actions
           The action space consist of a list in which each element
           corresponds to the desired speed that RL vehicles should travel in
           that region of space

       Rewards
           The reward is the outflow of the bottleneck plus a reward
           for RL vehicles making forward progress
    """

    def __init__(self, env_params, sumo_params, scenario):
        super().__init__(env_params, sumo_params, scenario)
        for p in ADDITIONAL_VSL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # default (edge, segment, controlled) status
        add_env_params = self.env_params.additional_params
        default = [("1", 1, True), ("2", 1, True), ("3", 1, True),
                   ("4", 1, True), ("5", 1, True)]
        super(DesiredVelocityEnv, self).__init__(env_params, sumo_params,
                                                 scenario)
        self.segments = add_env_params.get("controlled_segments", default)

        # number of segments for each edge
        self.num_segments = [segment[1] for segment in self.segments]

        # whether an edge is controlled
        self.is_controlled = [segment[2] for segment in self.segments]

        self.num_controlled_segments = [
            segment[1] for segment in self.segments if segment[2]
        ]

        # sum of segments
        self.total_segments = int(
            np.sum([segment[1] for segment in self.segments]))
        # sum of controlled segments
        segment_list = [segment[1] for segment in self.segments if segment[2]]
        self.total_controlled_segments = int(np.sum(segment_list))

        # list of controlled edges for comparison
        self.controlled_edges = [
            segment[0] for segment in self.segments if segment[2]
        ]

        additional_params = env_params.additional_params

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.slices = {}
        for edge, num_segments, _ in self.segments:
            edge_length = self.scenario.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments + 1)

        # get info for observed segments
        self.obs_segments = additional_params.get("observed_segments")

        # number of segments for each edge
        self.num_obs_segments = [segment[1] for segment in self.obs_segments]

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.obs_slices = {}
        for edge, num_segments in self.obs_segments:
            edge_length = self.scenario.edge_length(edge)
            self.obs_slices[edge] = np.linspace(0, edge_length,
                                                num_segments + 1)

        # self.symmetric is True if all lanes in a segment
        # have same action, else False
        self.symmetric = additional_params.get("symmetric")

        # action index tells us, given an edge and a lane,the offset into
        # rl_actions that we should take.
        self.action_index = [0]
        for i, (edge, segment, controlled) in enumerate(self.segments[:-1]):
            if self.symmetric:
                self.action_index += [
                    self.action_index[i] + segment * controlled
                ]
            else:
                num_lanes = self.scenario.num_lanes(edge)
                self.action_index += [
                    self.action_index[i] + segment * controlled * num_lanes
                ]

        self.action_index = {}
        action_list = [0]
        index = 0
        for (edge, num_segments, controlled) in self.segments:
            if controlled:
                if self.symmetric:
                    self.action_index[edge] = [action_list[index]]
                    action_list += [action_list[index] + controlled]
                else:
                    num_lanes = self.scenario.num_lanes(edge)
                    self.action_index[edge] = [action_list[index]]
                    action_list += [
                        action_list[index] +
                        num_segments * controlled * num_lanes
                    ]
                index += 1

    @property
    def observation_space(self):
        """See class definition."""
        num_obs = 0
        # density and velocity for rl and non-rl vehicles per segment
        # Last element is the outflow
        for segment in self.obs_segments:
            num_obs += 4 * segment[1] * self.scenario.num_lanes(segment[0])
        num_obs += 1
        return Box(low=0.0, high=1.0, shape=(num_obs, ), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        if self.symmetric:
            action_size = self.total_controlled_segments
        else:
            action_size = 0.0
            for segment in self.segments:  # iterate over segments
                if segment[2]:  # if controlled
                    num_lanes = self.scenario.num_lanes(segment[0])
                    action_size += num_lanes * segment[1]
        return Box(
            low=-1.5, high=1.0, shape=(int(action_size), ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # action space is number of vehicles in each segment in each lane,
        # number of rl vehicles in each segment in each lane
        # mean speed in each segment, and mean rl speed in each
        # segment in each lane
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
        NUM_VEHICLE_NORM = 20
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.scenario.num_lanes(edge)
            num_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            num_rl_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            rl_vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            ids = self.vehicles.get_ids_by_edge(edge)
            lane_list = self.vehicles.get_lane(ids)
            pos_list = self.vehicles.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.obs_slices[edge],
                                          pos_list[i]) - 1
                if id in self.vehicles.get_rl_ids():
                    rl_vehicle_speeds[segment, lane_list[i]] \
                        += self.vehicles.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] \
                        += self.vehicles.get_speed(id)
                    num_vehicles[segment, lane_list[i]] += 1

            # normalize

            num_vehicles /= NUM_VEHICLE_NORM
            num_rl_vehicles /= NUM_VEHICLE_NORM
            num_vehicles_list += num_vehicles.flatten().tolist()
            num_rl_vehicles_list += num_rl_vehicles.flatten().tolist()
            vehicle_speeds_list += vehicle_speeds.flatten().tolist()
            rl_speeds_list += rl_vehicle_speeds.flatten().tolist()

        unnorm_veh_list = np.asarray(num_vehicles_list) * \
            NUM_VEHICLE_NORM
        unnorm_rl_list = np.asarray(num_rl_vehicles_list) * \
            NUM_VEHICLE_NORM
        # compute the mean speed if the speed isn't zero
        num_rl = len(num_rl_vehicles_list)
        num_veh = len(num_vehicles_list)
        mean_speed = np.nan_to_num([
            vehicle_speeds_list[i] / unnorm_veh_list[i]
            if int(unnorm_veh_list[i]) else 0 for i in range(num_veh)
        ])
        mean_speed_norm = mean_speed / 50
        mean_rl_speed = np.nan_to_num([
            rl_speeds_list[i] / unnorm_rl_list[i]
            if int(unnorm_rl_list[i]) else 0 for i in range(num_rl)
        ]) / 50
        outflow = np.asarray(
            self.vehicles.get_outflow_rate(20 * self.sim_step) / 2000.0)
        return np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow]))

    def _apply_rl_actions(self, rl_actions):
        """
        RL actions are split up into 3 levels.
        First, they're split into edge actions.
        Then they're split into segment actions.
        Then they're split into lane actions.
        """
        for rl_id in self.vehicles.get_rl_ids():
            edge = self.vehicles.get_edge(rl_id)
            lane = self.vehicles.get_lane(rl_id)
            if edge:
                # If in outer lanes, on a controlled edge, in a controlled lane
                if edge[0] != ':' and edge in self.controlled_edges:
                    pos = self.vehicles.get_position(rl_id)

                    if not self.symmetric:
                        num_lanes = self.scenario.num_lanes(edge)
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[int(lane) + bucket * num_lanes +
                                            self.action_index[edge]]
                    else:
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[bucket + self.action_index[edge]]

                    traci_veh = self.traci_connection.vehicle
                    max_speed_curr = traci_veh.getMaxSpeed(rl_id)
                    next_max = np.clip(max_speed_curr + action, 0.01, 23.0)
                    traci_veh.setMaxSpeed(rl_id, next_max)

                else:
                    # set the desired velocity of the controller to the default
                    self.traci_connection.vehicle.setMaxSpeed(rl_id, 23.0)

    def compute_reward(self, state, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""

        if self.env_params.evaluate:
            if self.time_counter == self.env_params.horizon:
                reward = self.vehicles.get_outflow_rate(500)
            else:
                return 0
        else:
            reward = self.vehicles.get_outflow_rate(10 * self.sim_step) / \
                     (2000.0 * self.scaling)
        return reward

    def reset(self):
        add_params = self.env_params.additional_params
        if add_params.get("reset_inflow"):
            inflow_range = add_params.get("inflow_range")
            flow_rate = np.random.uniform(
                min(inflow_range), max(inflow_range)) * self.scaling
            for _ in range(100):
                try:
                    inflow = InFlows()
                    inflow.add(
                        veh_type="followerstopper",
                        edge="1",
                        vehs_per_hour=flow_rate * .1,
                        departLane="random",
                        departSpeed=10)
                    inflow.add(
                        veh_type="human",
                        edge="1",
                        vehs_per_hour=flow_rate * .9,
                        departLane="random",
                        departSpeed=10)

                    additional_net_params = {"scaling": self.scaling}
                    net_params = NetParams(
                        inflows=inflow,
                        no_internal_links=False,
                        additional_params=additional_net_params)

                    vehicles = Vehicles()
                    vehicles.add(
                        veh_id="human",
                        speed_mode=9,
                        lane_change_controller=(SumoLaneChangeController, {}),
                        routing_controller=(ContinuousRouter, {}),
                        lane_change_mode=0,  # 1621,#0b100000101,
                        num_vehicles=1 * self.scaling)
                    vehicles.add(
                        veh_id="followerstopper",
                        acceleration_controller=(RLController, {}),
                        lane_change_controller=(SumoLaneChangeController, {}),
                        routing_controller=(ContinuousRouter, {}),
                        speed_mode=9,
                        lane_change_mode=0,
                        num_vehicles=1 * self.scaling)
                    self.vehicles = vehicles

                    # delete the cfg and net files
                    net_path = self.scenario.generator.net_path
                    net_name = net_path + self.scenario.generator.name
                    cfg_path = self.scenario.generator.cfg_path
                    cfg_name = cfg_path + self.scenario.generator.name
                    for f in glob.glob(net_name + '*'):
                        os.remove(f)
                    for f in glob.glob(cfg_name + '*'):
                        os.remove(f)

                    self.scenario = self.scenario.__class__(
                        name=self.scenario.orig_name,
                        generator_class=self.scenario.generator_class,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.scenario.initial_config,
                        traffic_lights=self.scenario.traffic_lights)
                    observation = super().reset()

                    # reset the timer to zero
                    self.time_counter = 0

                    return observation

                except Exception as e:
                    print('error on reset ', e)
                    # perform the generic reset function

        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation
