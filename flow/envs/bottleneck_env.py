from flow.envs.lane_changing import LaneChangeAccelEnv
from flow.core import rewards
from flow.core import multi_agent_rewards
from flow.controllers.velocity_controllers import FollowerStopper
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from collections import defaultdict
from copy import deepcopy

import numpy as np

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

ENV_PARAMS = [
    "disable_tb",  # whether the toll booth should be active
    "disable_ramp_metering",  # whether the ramp meter is active
    "target_velocity",  # velocity to use in reward functions
    "num_steps",  # horizon
    "add_rl_if_exit",  # if an RL vehicle exits, place it back at the front
    "segments", # number of regions for velocity bottleneck controller
]

NET_PARAMS = [
    "scaling",  # the factor multiplying number of lanes.
]

START_RECORD_TIME = 0.0
PERIOD = 10.0


class BridgeTollEnv(LaneChangeAccelEnv):
    def __init__(self, env_params, sumo_params, scenario):
        """Environment used as a simplified representation of the toll
            booth portion of the bay bridge. Contains ramp meters,
            and a toll both.

           Additional
           ----------
           Vehicles are rerouted to the start of their original routes
           once they reach the end of the network in order
           to ensure a constant number of vehicles.
           """
        for k in env_params.additional_params.keys():
            if k not in ENV_PARAMS:
                raise Exception(
                    "Unknown env params key `{}`, all keys: {}".format(
                        k, ENV_PARAMS))

        for k in scenario.net_params.additional_params.keys():
            if k not in NET_PARAMS:
                raise Exception(
                    "Unknown net params key `{}`, all keys: {}".format(
                        k, NET_PARAMS))

        self.num_rl = deepcopy(scenario.vehicles.num_rl_vehicles)
        super().__init__(env_params, sumo_params, scenario)
        # tells how scaled the number of lanes are
        self.scaling = scenario.net_params.additional_params.get("scaling", 1)
        self.edge_dict = defaultdict(list)
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES * self.scaling))
        # these values place the fast track in the middle lanes
        self.fast_track_lanes = range(int(np.ceil(1.5 * self.scaling)),
                                      int(np.ceil(2.6 * self.scaling)))
        self.tl_state = ""
        self.disable_tb = False
        self.disable_ramp_metering = False
        self.add_rl_if_exit = False
        self.rl_id_list = deepcopy(self.vehicles.get_rl_ids())

        self.next_period = START_RECORD_TIME / self.sim_step
        self.cars_arrived = 0

        print(env_params.additional_params)

        if "disable_tb" in env_params.additional_params:
            self.disable_tb = env_params.get_additional_param("disable_tb")

        if "disable_ramp_metering" in env_params.additional_params:
            self.disable_ramp_metering = \
                env_params.get_additional_param("disable_ramp_metering")

        if "add_rl_if_exit" in env_params.additional_params:
            self.add_rl_if_exit = \
                env_params.get_additional_param("add_rl_if_exit")

    def additional_command(self):
        super().additional_command()
        # build a list of vehicles and their edges and positions
        self.edge_dict = defaultdict(list)
        # update the dict with all the edges in edge_list
        # so we can look forward for edges
        self.edge_dict.update(
            (k, [[] for _ in range(MAX_LANES * self.scaling)])
            for k in EDGE_LIST)
        for veh_id in self.vehicles.get_ids():
            try:
                edge = self.vehicles.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict.update({edge: [[] for _ in range(MAX_LANES * self.scaling)]})
                lane = self.vehicles.get_lane(veh_id)  # integer
                pos = self.vehicles.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except:
                pass
        if not self.disable_tb:
            self.apply_toll_bridge_control()
        if not self.disable_ramp_metering:
            self.ramp_meter_lane_change_control()

        if self.time_counter > self.next_period:
            self.density = self.cars_arrived  # / (PERIOD/self.sim_step)
            # print(self.density)
            self.next_period += PERIOD / self.sim_step
            self.cars_arrived = 0

        self.cars_arrived += self.vehicles.num_arrived

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
                        # Disable lane changes inside Toll Area
                        lane_change_mode = \
                            self.vehicles.get_lane_change_mode(veh_id)
                        color = self.traci_connection.vehicle.getColor(veh_id)
                        self.cars_before_ramp[veh_id] = {"lane_change_mode":
                                                             lane_change_mode,
                                                         "color":
                                                             color}
                        self.traci_connection.vehicle.setLaneChangeMode(
                            veh_id, 512)
                        self.traci_connection.vehicle.setColor(
                            veh_id, (0, 255, 255, 0))

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
                        0, np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL /
                                            self.sim_step,
                                            1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(
                        0, np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK
                                            / self.sim_step,
                                            1 / self.sim_step))

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
                        self.traci_connection.vehicle.setLaneChangeMode(veh_id,
                                                                        512)
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
            self.traci_connection.trafficlights.setRedYellowGreenState(
                tlsID=TB_TL_ID, state=newTLState)

    def distance_to_bottlneck(self, veh_id):
        pre_bottleneck_edges = {str(i): self.scenario.edge_length(str(i)) for i in [1, 2, 3]}
        edge_pos = self.vehicles.get_position(veh_id)
        edge = self.vehicles.get_edge(veh_id)
        if edge in pre_bottleneck_edges:
            total_length = pre_bottleneck_edges[edge] - edge_pos
            for next_edge in range(int(edge)+1, 4):
                total_length+=pre_bottleneck_edges[str(next_edge)]
            return total_length
        else:
            return -1

    def get_bottleneck_outflow_vehicles_per_hour(self, sample_period):
        return self.vehicles.get_outflow_rate(sample_period)

    def get_bottleneck_density(self, lanes=None):
        BOTTLE_NECK_LEN = 280
        if lanes:
            print(self.vehicles.get_ids_by_edge(['3', '4']))
            veh_ids = [veh_id for veh_id in self.vehicles.get_ids_by_edge(['3', '4']) if str(self.vehicles.get_edge(veh_id))+ "_" + str(self.vehicles.get_lane(veh_id)) in lanes]
            print("======================", veh_ids)
        else:
            veh_ids = self.vehicles.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

    def get_avg_bottleneck_velocity(self):
        veh_ids = self.vehicles.get_ids_by_edge(['3', '4', '5'])
        return sum(self.vehicles.get_speed(veh_ids))/len(veh_ids) if len(veh_ids) != 0 else 0

class BottleNeckEnv(BridgeTollEnv):
    """Environment used to train vehicles to effectively
        pass through a bottleneck.

       States
       ------
       An observation is the edge position, speed, lane, and edge number of the
       AV, the distance to and velocity of the vehicles
       in front and behind the AV for all lanes. Additionally, we pass the
       density and average velocity of all edges. Finally, we pad with zeros
       in case an AV has exited the system.
       Note: the vehicles are arranged in an initial order, so we pad
       the missing vehicle at its normal position in the order

       Actions
       -------
       The action space consist of a list in which the first half
       is accelerations and the second half is a direction for lane changing
       that we round

       Rewards
       -------
       The reward is the two-norm of the difference between the speed of all
       vehicles in the network and some desired speed. To this we add
       a positive reward for moving the vehicles forward

       Termination
       -----------
       A rollout is terminated once the time horizon is reached.

       """

    @property
    def observation_space(self):
        num_edges = len(self.scenario.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling \
                  * num_rl_veh + 4 * num_rl_veh
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")
        print(num_obs)
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")
        return Box(low=-float("inf"), high=float("inf"), shape=(num_obs,),
                   dtype=np.float32)

    def get_state(self):

        headway_scale = 1000

        rl_ids = self.vehicles.get_rl_ids()

        # rl vehicle data (absolute position, speed, and lane index)
        rl_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list.index(veh_id)
            if rl_id_num != id_counter:
                rl_obs = np.concatenate((rl_obs,
                                         np.zeros(4 * (rl_id_num -
                                                       id_counter))))
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
            rl_obs = np.concatenate((rl_obs,
                                     [self.get_x_by_id(veh_id) / 1000,
                                      (self.vehicles.get_speed(veh_id) /
                                       self.max_speed),
                                      (self.vehicles.get_lane(veh_id) /
                                       MAX_LANES),
                                      edge_num]))
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
                pad_mat = np.zeros(4 * MAX_LANES * self.scaling *
                                   (rl_id_num - id_counter))
                relative_obs = np.concatenate((relative_obs, pad_mat))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1
            num_lanes = MAX_LANES * self.scaling
            headway = np.asarray([1000 for _ in
                                  range(num_lanes)]) / headway_scale
            tailway = np.asarray([1000 for _ in
                                  range(num_lanes)]) / headway_scale
            vel_in_front = np.asarray([0 for _ in
                                       range(num_lanes)]) / self.max_speed
            vel_behind = np.asarray([0 for _ in
                                     range(num_lanes)]) / self.max_speed

            lane_leaders = self.vehicles.get_lane_leaders(veh_id)
            lane_followers = self.vehicles.get_lane_followers(veh_id)
            lane_headways = self.vehicles.get_lane_headways(veh_id)
            lane_tailways = self.vehicles.get_lane_tailways(veh_id)
            headway[0:len(lane_headways)] = (np.asarray(lane_headways) /
                                             headway_scale)
            tailway[0:len(lane_tailways)] = (np.asarray(lane_tailways) /
                                             headway_scale)
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (self.vehicles.get_speed(lane_leader) /
                                       self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = (self.vehicles.get_speed(lane_follower)
                                     / self.max_speed)

            relative_obs = np.concatenate((relative_obs, headway,
                                           tailway, vel_in_front, vel_behind))

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
                avg_speed = (sum(self.vehicles.get_speed(veh_ids))
                             / len(veh_ids)) / self.max_speed
                density = len(veh_ids) / self.scenario.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        return np.concatenate((rl_obs, relative_obs, edge_obs))

    def compute_reward(self, state, rl_actions, **kwargs):
        num_rl = self.vehicles.num_rl_vehicles
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return (rewards.desired_velocity(self) +
                rewards.rl_forward_progress(self, gain=0.1) -
                rewards.boolean_action_penalty(lane_change_acts, gain=1.0))

    def sort_by_position(self):
        if self.env_params.sort_vehicles:
            sorted_ids = sorted(self.vehicles.get_ids(),
                                key=self.get_x_by_id)
            return sorted_ids, None
        else:
            return self.vehicles.get_ids(), None

    def _apply_rl_actions(self, actions):
        """
        See parent class

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

    def additional_command(self):
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.vehicles.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(set(self.rl_id_list).difference(
                self.vehicles.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                # FIXME(ev) the try is for when we've already
                # FIXME called to introduce
                # FIXME but the introduce has been blocked by an inflow
                # FIXME a better way would be keeping track of when
                # FIXME we have made this call
                try:
                    self.traci_connection.vehicle.addFull(
                        rl_id, 'route1', typeID=str('rl'),
                        departLane=str(lane_num),
                        departPos="0", departSpeed="max")
                except:
                    pass


class DesiredVelocityEnv(BridgeTollEnv):
    """Environment used to train vehicles to effectively pass through a bottleneck
       by specifying the velocity that RL vehicles should attempt to travel
       in certain regions of space

       States
       ------
       An observation is the number of vehicles in each lane in each
       segment

       Actions
       -------
       The action space consist of a list in which each element
       corresponds to the desired speed that RL vehicles should travel in that
       region of space

       Rewards
       -------
       The reward is the outflow of the bottleneck plus a reward
       for RL vehicles making forward progress
    """

    def __init__(self, env_params, sumo_params, scenario):

        default = [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1)]
        super(DesiredVelocityEnv, self).__init__(env_params, sumo_params, scenario)
        self.segments = self.env_params.additional_params.get("segments", default)
        self.num_segments = [segment[1] for segment in self.segments]
        self.controlled_edges = [segment[0] for segment in self.segments]
        self.total_segments = int(np.sum([segment[1] for segment in self.segments]))
        # for convenience, construct the relevant positions we are looking for
        self.slices = {}
        for edge, num_segments in self.segments:
            edge_length = self.scenario.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments)

        # construct an indexing to be used for figuring out which
        # action is useful
        self.action_index = [0]
        for i, segment in enumerate(self.num_segments[:-1]):
            self.action_index += [self.action_index[i] + segment]

    @property
    def observation_space(self):
        num_obs = 0
        for segment in self.segments:
            num_obs += segment[1] * self.scenario.num_lanes(segment[0])
        num_obs += self.total_segments
        return Box(low=-float("inf"), high=float("inf"), shape=(num_obs,),
                   dtype=np.float32)

    @property
    def action_space(self):
        return Box(low=5.0, high=self.max_speed, shape=(int(self.total_segments),
                                                      ), dtype=np.float32)

    def get_state(self):
        # FIXME(ev) add information about AVs)
        num_vehicles_list = []
        segment_speeds = np.zeros((self.total_segments, 2))
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.scenario.num_lanes(edge)
            num_vehicles = np.zeros((self.num_segments[i], num_lanes))
            ids = self.vehicles.get_ids_by_edge(edge)
            lane_list = self.vehicles.get_lane(ids)
            pos_list = self.vehicles.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.slices[edge], pos_list[i]) - 1
                num_vehicles[segment, lane_list[i]] += 1
                segment_speeds[segment][0] = self.vehicles.get_speed(id)
                segment_speeds[segment][1] += 1

            # normalize
            num_vehicles /= 20
            num_vehicles_list += num_vehicles.flatten().tolist()
        mean_speed = np.nan_to_num([segment_speeds[i][0]/segment_speeds[i][1]
                      for i in range(self.total_segments)])/50
        return np.concatenate((num_vehicles_list, mean_speed))

    def _apply_rl_actions(self, actions):
        rl_actions = (20*actions).clip(self.action_space.low, self.action_space.high)
        # FIXME(ev) make it so that you don't have to control everrrry edge
        veh_ids = [veh_id for veh_id in self.vehicles.get_ids()
                   if isinstance(self.vehicles.get_acc_controller(veh_id), FollowerStopper)]
        for rl_id in veh_ids:
            edge = self.vehicles.get_edge(rl_id)
            if edge:
                if edge[0] != ':' and edge in self.controlled_edges:
                    pos = self.vehicles.get_position(rl_id)
                    # find what segment we fall into
                    bucket = np.searchsorted(self.slices[edge], pos) - 1
                    action = rl_actions[bucket + self.action_index[int(edge) - 1]]
                    # set the desired velocity of the controller to the action
                    controller = self.vehicles.get_acc_controller(rl_id)
                    controller.v_des = action

    def compute_reward(self, state, rl_actions, **kwargs):
        reward = self.vehicles.get_outflow_rate(20*self.sim_step)/3600.0 + \
            0.01*rewards.desired_velocity(self)/self.max_speed
        return reward


class MultiBottleNeckEnv(BottleNeckEnv):
    """Multiagent environment used to train vehicles
    to effectively pass through a bottleneck.

       States
       ------
       An observation is the edge position, speed, lane, and edge number of the
       AV, the distance to and velocity of the vehicles
       in front and behind the AV for all lanes. Additionally, we pass the
       density and average velocity of all edges. Finally, we add the
       position and velocity of all other AVs in order of the list
       Additionally, we pad with zeros
       in case an AV has exited the system.
       Note: the vehicles are arranged in an initial order, so we pad
       the missing vehicle at its normal position in the order

       Actions
       -------
       The action space consist of a list of lists in which
       the first half of each list
       is accelerations and the second half is a direction for lane changing
       that we round

       Rewards
       -------
       The reward is the two-norm of the difference between the speed of all
       vehicles in the network and some desired speed. To this we add
       a positive reward for moving the vehicles forward and a
       penalty for lane changing

       Termination
       -----------
       A rollout is terminated once the time horizon is reached.

       """

    @property
    def observation_space(self):
        num_edges = len(self.scenario.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling + 4 * num_rl_veh
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")
        print(num_obs)
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")
        return Tuple(tuple(Box(low=-float("inf"), high=float("inf"),
                               shape=(num_obs,), dtype=np.float32)
                           for _ in range(self.num_rl)))

    @property
    def action_space(self):
        return [self.lane_change_action_space() for _ in range(self.num_rl)]

    def lane_change_action_space(self):
        """
        See parent class
        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) lane-change action from -1 to 1,
           used to determine the lateral direction the vehicle will take.
        """
        max_decel = self.env_params.max_decel
        max_accel = self.env_params.max_accel

        lb = [-abs(max_decel), -1]
        ub = [max_accel, 1]

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def get_state(self):

        headway_scale = 1000

        rl_ids = self.rl_id_list
        obs_list = []
        for veh_id in rl_ids:
            rl_obs = np.empty(0)
            # if the vehicle is in the system, look for it
            if veh_id in self.vehicles.get_rl_ids():
                # rearrange both rl_id_list and rl_ids to have the right order
                # maximal number of ids
                all_rl_ids = deepcopy(rl_ids)
                all_index = all_rl_ids.index(veh_id)
                # list with possibly missing ids
                local_ids = deepcopy(self.vehicles.get_rl_ids())
                local_index = local_ids.index(veh_id)
                del all_rl_ids[all_index]
                del local_ids[local_index]
                all_rl_ids = [veh_id] + all_rl_ids
                local_ids = [veh_id] + local_ids

                # Get the info for for all the vehicles
                id_counter = 0
                for av_id in local_ids:
                    # check if we have skipped a vehicle, if not, pad
                    rl_id_num = all_rl_ids.index(av_id)
                    if rl_id_num != id_counter:
                        rl_obs = np.concatenate((rl_obs,
                                                 np.zeros(4 * (rl_id_num -
                                                               id_counter))))
                        id_counter = rl_id_num + 1
                    else:
                        id_counter += 1
                        rl_obs = np.concatenate((rl_obs,
                                                 self.get_vehicle_info(av_id)))
                # if all the missing vehicles are at the end, pad
                diff = self.num_rl - int(rl_obs.shape[0] / 4)
                if diff > 0:
                    rl_obs = np.concatenate((rl_obs, np.zeros(4 * diff)))

                # relative vehicles data (lane headways,
                # tailways, vel_ahead, and vel_behind)
                num_lanes = MAX_LANES * self.scaling
                headway = np.asarray([1000 for _ in
                                      range(num_lanes)]) / headway_scale
                tailway = np.asarray([1000 for _ in
                                      range(num_lanes)]) / headway_scale
                vel_in_front = np.asarray([0 for _ in
                                           range(num_lanes)]) / self.max_speed
                vel_behind = np.asarray([0 for _ in
                                         range(num_lanes)]) / self.max_speed

                lane_leaders = self.vehicles.get_lane_leaders(veh_id)
                lane_followers = self.vehicles.get_lane_followers(veh_id)
                lane_headways = self.vehicles.get_lane_headways(veh_id)
                lane_tailways = self.vehicles.get_lane_tailways(veh_id)
                headway[0:len(lane_headways)] = (np.asarray(lane_headways) /
                                                 headway_scale)
                tailway[0:len(lane_tailways)] = (np.asarray(lane_tailways) /
                                                 headway_scale)
                for i, lane_leader in enumerate(lane_leaders):
                    if lane_leader != '':
                        vel_in_front[i] = (self.vehicles.get_speed(lane_leader)
                                           / self.max_speed)
                for i, lane_follower in enumerate(lane_followers):
                    if lane_followers != '':
                        vel_behind[i] = (self.vehicles.get_speed(lane_follower)
                                         / self.max_speed)

                relative_obs = np.concatenate((headway, tailway,
                                               vel_in_front, vel_behind))
            # otherwise, just pass zeros
            else:
                rl_obs = np.zeros(4 * self.num_rl)
                relative_obs = np.zeros(4 * MAX_LANES * self.scaling)

            # per edge data (average speed, density
            edge_obs = []
            for edge in self.scenario.get_edge_list():
                veh_ids = self.vehicles.get_ids_by_edge(edge)
                if len(veh_ids) > 0:
                    avg_speed = (sum(self.vehicles.get_speed(veh_ids))
                                 / len(veh_ids)) / self.max_speed
                    density = len(veh_ids) / self.scenario.edge_length(edge)
                    edge_obs += [avg_speed, density]
                else:
                    edge_obs += [0, 0]

            obs_list.append(np.concatenate((rl_obs, relative_obs,
                                            edge_obs)))

        return obs_list

    def compute_reward(self, state, rl_actions, **kwargs):
        num_rl = self.num_rl
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return rewards.max_edge_velocity(self, ["4", "5"]) + \
               rewards.rl_forward_progress(self, gain=0.1) - \
               rewards.boolean_action_penalty(lane_change_acts, gain=1.0)

    def _apply_rl_actions(self, rl_actions):
        """
        See parent class

        Apply a lane change for a multi-agent system
        """
        accelerations = []
        directions = []
        for veh_id in self.vehicles.get_rl_ids():
            index = self.rl_id_list.index(veh_id)
            accelerations.append(rl_actions[index][0][0])
            direction = np.round(rl_actions[index][0][1]).clip(min=-1, max=1)

            # if we have lane changed recently or are on a junction
            try:
                if self.time_counter <= self.lane_change_duration + \
                        self.vehicles.get_state(veh_id, 'last_lc') or \
                        self.vehicles.get_edge(veh_id)[0] == ':':
                    direction = 0
            except:
                direction = 0
            directions.append(direction)

        self.apply_acceleration(self.vehicles.get_rl_ids(),
                                acc=accelerations)
        self.apply_lane_change(self.vehicles.get_rl_ids(),
                               direction=directions)

    # ===============================
    # ============ UTILS ============
    # ===============================
    def get_vehicle_info(self, veh_id):
        """
        Gets standard information for the state space
        :return: list containing edge num, edge position, speed, and lane
        """
        edge_num = self.vehicles.get_edge(veh_id)
        if edge_num is None:
            edge_num = -1
        elif edge_num == '':
            edge_num = -1
        elif edge_num[0] == ':':
            edge_num = -1
        else:
            edge_num = int(edge_num) / 6
        veh_obs = [self.get_x_by_id(veh_id) / 1000,
                   self.vehicles.get_speed(veh_id) / self.max_speed,
                   self.vehicles.get_lane(veh_id) / MAX_LANES,
                   edge_num / 5]
        return veh_obs
