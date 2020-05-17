"""
Environments for training vehicles to reduce capacity drops in a bottleneck.

This environment was used in:

E. Vinitsky, K. Parvate, A. Kreidieh, C. Wu, Z. Hu, A. Bayen, "Lagrangian
Control through Deep-RL: Applications to Bottleneck Decongestion," IEEE
Intelligent Transportation Systems Conference (ITSC), 2018.
"""

from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import InFlows, NetParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams

from copy import deepcopy

import numpy as np
from gym.spaces.box import Box

from flow.core import rewards
from flow.envs.base import Env

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth
EDGE_BEFORE_TOLL = "1"  # Specifies which edge number is before toll booth
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"  # Specifies which edge number is after toll booth
NUM_TOLL_LANES = MAX_LANES

TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # how close for the ramp meter to start going off

EDGE_BEFORE_RAMP_METER = "2"  # Specifies which edge is before ramp meter
EDGE_AFTER_RAMP_METER = "3"  # Specifies which edge is after ramp meter
NUM_RAMP_METERS = MAX_LANES

RAMP_METER_AREA = 80  # Area occupied by ramp meter

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3  # Average waiting time at fast track
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15  # Average waiting time at toll

BOTTLE_NECK_LEN = 280  # Length of bottleneck
NUM_VEHICLE_NORM = 20

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

START_RECORD_TIME = 0.0  # Time to start recording
PERIOD = 10.0


class BottleneckEnv(Env):
    """Abstract bottleneck environment.

    This environment is used as a simplified representation of the toll booth
    portion of the bay bridge. Contains ramp meters, and a toll both.

    Additional
    ----------
        Vehicles are rerouted to the start of their original routes once
        they reach the end of the network in order to ensure a constant
        number of vehicles.

    Attributes
    ----------
    scaling : int
        A factor describing how many lanes are in the system. Scaling=1 implies
        4 lanes going to 2 going to 1, scaling=2 implies 8 lanes going to 4
        going to 2, etc.
    edge_dict : dict of dicts
        A dict mapping edges to a dict of lanes where each entry in the lane
        dict tracks the vehicles that are in that lane. Used to save on
        unnecessary lookups.
    cars_waiting_for_toll : {veh_id: {lane_change_mode: int, color: (int)}}
        A dict mapping vehicle ids to a dict tracking the color and lane change
        mode of vehicles before they entered the toll area. When vehicles exit
        the tollbooth area, these values are used to set the lane change mode
        and color of the vehicle back to how they were before they entered the
        toll area.
    cars_before_ramp : {veh_id: {lane_change_mode: int, color: (int)}}
        Identical to cars_waiting_for_toll, but used to track cars approaching
        the ramp meter versus approaching the tollbooth.
    toll_wait_time : np.ndarray(float)
        Random value, sampled from a gaussian indicating how much a vehicle in
        each lane should wait to pass through the toll area. This value is
        re-sampled for each approaching vehicle. That is, whenever a vehicle
        approaches the toll area, we re-sample from the Gaussian to determine
        its weight time.
    fast_track_lanes : np.ndarray(int)
        Middle lanes of the tollbooth are declared fast-track lanes, this numpy
        array keeps track of which lanes these are. At a fast track lane, the
        mean of the Gaussian from which we sample wait times is given by
        MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK.
    tl_state : str
        String tracking the color of the traffic lights at the tollbooth. These
        traffic lights are used imitate the effect of a tollbooth. If lane 1-4
        are respectively green, red, red, green, then this string would be
        "GrrG"
    n_crit : int
        The ALINEA algorithm adjusts the ratio of red to green time for the
        ramp-metering traffic light based on feedback on how congested the
        system is. As the measure of congestion, we use the number of vehicles
        stuck in the bottleneck (edge 4). The critical operating value it tries
        to stabilize the number of vehicles in edge 4 is n_crit. If there are
        more than n_crit vehicles on edge 4, we increase the fraction of red
        time to decrease the inflow to edge 4.
    q_max : float
        The ALINEA algorithm tries to control the flow rate through the ramp
        meter. q_max is the maximum possible estimated flow we allow through
        the bottleneck and can be converted into a maximum value for the ratio
        of green to red time that we allow.
    q_min : float
        Similar to q_max, this is used to set the minimum value of green to red
        ratio that we allow.
    q : float
        This value tracks the flow we intend to allow through the bottleneck.
        For details on how it is computed, please read the alinea method or the
        paper linked in that method.
    feedback_update_time : float
        The parameters of the ALINEA algorithm are only updated every
        feedback_update_time seconds.
    feedback_timer : float
        This keeps track of how many seconds have passed since the ALINEA
        parameters were last updated. If it exceeds feedback_update_time, the
        parameters are updated
    cycle_time : int
        This value tracks how long a green-red cycle of the ramp meter is. The
        first self.green_time seconds will be green and the remainder of the
        cycle will be red.
    ramp_state : np.ndarray
        Array of floats of length equal to the number of lanes. For each lane,
        this value tracks how many seconds of a given cycle have passed in that
        lane. Each lane is offset from its adjacent lanes by
        cycle_offset/(self.scaling * MAX_LANES) seconds. This offsetting means
        that lights are offset in when they releasse vehicles into the
        bottleneck. This helps maximize the throughput of the ramp meter.
    green_time : float
        How many seconds of a given cycle the light should remain green 4.
        Defaults to 4 as this is just enough time for two vehicles to enter the
        bottleneck from a given traffic light.
    feedback_coeff : float
        This is the gain on the feedback in the ALINEA algorithm
    smoothed_num : np.ndarray
        Numpy array keeping track of how many vehicles were in edge 4 over the
        last 10 time seconds. This provides a more stable estimate of the
        number of vehicles in edge 4.
    outflow_index : int
        Keeps track of which index of smoothed_num we should update with the
        latest number of vehicles in the bottleneck. Should eventually be
        deprecated as smoothed_num should be switched to a queue instead of an
        array.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the BottleneckEnv class."""
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        env_add_params = self.env_params.additional_params
        # tells how scaled the number of lanes are
        self.scaling = network.net_params.additional_params.get("scaling", 1)
        self.edge_dict = dict()
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES * self.scaling))
        self.fast_track_lanes = range(
            int(np.ceil(1.5 * self.scaling)), int(np.ceil(2.6 * self.scaling)))

        self.tl_state = ""

        # values for the ALINEA ramp meter algorithm
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
        self.feedback_coeff = env_add_params.get("feedback_coeff", 20)

        self.smoothed_num = np.zeros(10)  # averaged number of vehs in '4'
        self.outflow_index = 0

    def additional_command(self):
        """Build a dict with vehicle information.

        The dict contains the list of vehicles and their position for each edge
        and for each edge within the edge.
        """
        super().additional_command()

        # build a dict containing the list of vehicles and their position for
        # each edge and for each lane within the edge
        empty_edge = [[] for _ in range(MAX_LANES * self.scaling)]

        self.edge_dict = {k: deepcopy(empty_edge) for k in EDGE_LIST}
        for veh_id in self.k.vehicle.get_ids():
            try:
                edge = self.k.vehicle.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict[edge] = deepcopy(empty_edge)
                lane = self.k.vehicle.get_lane(veh_id)  # integer
                pos = self.k.vehicle.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except Exception:
                pass

        if not self.env_params.additional_params['disable_tb']:
            self.apply_toll_bridge_control()
        if not self.env_params.additional_params['disable_ramp_metering']:
            self.ramp_meter_lane_change_control()
            self.alinea()

        # compute the outflow
        veh_ids = self.k.vehicle.get_ids_by_edge('4')
        self.smoothed_num[self.outflow_index] = len(veh_ids)
        self.outflow_index = \
            (self.outflow_index + 1) % self.smoothed_num.shape[0]

    def ramp_meter_lane_change_control(self):
        """Control lane change behavior of vehicles near the ramp meters.

        If the traffic lights after the toll booth are enabled
        ('disable_ramp_metering' is False), we want to change the lane changing
        behavior of vehicles approaching the lights so that they stop changing
        lanes. This method disables their lane changing before the light and
        re-enables it after they have passed the light.

        Additionally, to visually make it clearer that the lane changing
        behavior of the vehicles has been adjusted, we temporary set the color
        of the affected vehicles to light blue.
        """
        cars_that_have_left = []
        for veh_id in self.cars_before_ramp:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                color = self.cars_before_ramp[veh_id]['color']
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = self.cars_before_ramp[veh_id][
                        'lane_change_mode']
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            del self.cars_before_ramp[veh_id]

        for lane in range(NUM_RAMP_METERS * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for veh_id, pos in cars_in_lane:
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        if self.simulator == 'traci':
                            # Disable lane changes inside Toll Area
                            lane_change_mode = \
                                self.k.kernel_api.vehicle.getLaneChangeMode(
                                    veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (0, 255, 255))
                        self.cars_before_ramp[veh_id] = {
                            'lane_change_mode': lane_change_mode,
                            'color': color
                        }

    def alinea(self):
        """Utilize the ALINEA algorithm for toll booth metering control.

        This acts as an implementation of the ramp metering control algorithm
        from the article:

        Spiliopoulou, Anastasia D., Ioannis Papamichail, and Markos
        Papageorgiou. "Toll plaza merging traffic control for throughput
        maximization." Journal of Transportation Engineering 136.1 (2009):
        67-76.

        Essentially, we apply feedback control around the value self.n_crit.
        We keep track of the number of vehicles in edge 4, average them across
        time ot get a smoothed value and then compute
        q_{t+1} = clip(q_t + K * (n_crit - n_avg), q_min, q_max). We then
        convert this into a cycle_time value via cycle_time = 7200 / q.
        Cycle_time = self.green_time + red_time i.e. the first self.green_time
        seconds of a cycle will be green, and the remainder will be all red.
        """
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
        self.k.traffic_light.set_state('3', ''.join(colors))

    def apply_toll_bridge_control(self):
        """Apply control to the toll bridge.

        Vehicles approaching the toll region slow down and stop lane changing.

        If 'disable_tb' is set to False, vehicles within TOLL_BOOTH_AREA of the
        end of edge EDGE_BEFORE_TOLL are labelled as approaching the toll
        booth. Their color changes and their lane changing is disabled. To
        force them to slow down/mimic the effect of the toll booth, we sample
        from a random normal distribution with mean
        MEAN_NUM_SECONDS_WAIT_AT_TOLL and std-dev 1/self.sim_step to get how
        long a vehicle should wait. We then turn on a red light for that many
        seconds.
        """
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.k.vehicle.get_lane(veh_id)
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = \
                        self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                    self.k.kernel_api.vehicle.setLaneChangeMode(
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
            del self.cars_waiting_for_toll[veh_id]

        traffic_light_states = ["G"] * NUM_TOLL_LANES * self.scaling

        for lane in range(NUM_TOLL_LANES * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_TOLL][lane]

            for veh_id, pos in cars_in_lane:
                if pos > TOLL_BOOTH_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        if self.simulator == 'traci':
                            lane_change_mode = self.k.kernel_api.vehicle.\
                                getLaneChangeMode(veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (255, 0, 255))
                        self.cars_waiting_for_toll[veh_id] = \
                            {'lane_change_mode': lane_change_mode,
                             'color': color}
                    else:
                        if pos > 50:
                            if self.toll_wait_time[lane] < 0:
                                traffic_light_states[lane] = "G"
                            else:
                                traffic_light_states[lane] = "r"
                                self.toll_wait_time[lane] -= 1

        new_tl_state = "".join(traffic_light_states)

        if new_tl_state != self.tl_state:
            self.tl_state = new_tl_state
            self.k.traffic_light.set_state(
                node_id=TB_TL_ID, state=new_tl_state)

    def get_bottleneck_density(self, lanes=None):
        """Return the density of specified lanes.

        If no lanes are specified, this function calculates the
        density of all vehicles on all lanes of the bottleneck edges.
        """
        bottleneck_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        if lanes:
            veh_ids = [
                veh_id for veh_id in bottleneck_ids
                if str(self.k.vehicle.get_edge(veh_id)) + "_" +
                str(self.k.vehicle.get_lane(veh_id)) in lanes
            ]
        else:
            veh_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

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

    def _apply_rl_actions(self, rl_actions):
        """See parent class.

        To be implemented by child classes.
        """
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / \
            (2000.0 * self.scaling)
        return reward

    def get_state(self):
        """See class definition."""
        return np.asarray([1])


class BottleneckAccelEnv(BottleneckEnv):
    """BottleneckAccelEnv.

    Environment used to train vehicles to effectively pass through a
    bottleneck.

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
        a positive reward for moving the vehicles forward, and a penalty to
        vehicles that lane changing too frequently.

    Termination
        A rollout is terminated once the time horizon is reached.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize BottleneckAccelEnv."""
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        self.max_speed = self.k.network.max_speed()

    @property
    def observation_space(self):
        """See class definition."""
        num_edges = len(self.k.network.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling \
            * num_rl_veh + 4 * num_rl_veh

        return Box(low=0, high=1, shape=(num_obs, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        headway_scale = 1000

        rl_ids = self.k.vehicle.get_rl_ids()

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
            edge_num = self.k.vehicle.get_edge(veh_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = int(edge_num) / 6
            rl_obs = np.concatenate((rl_obs, [
                self.k.vehicle.get_x_by_id(veh_id) / 1000,
                (self.k.vehicle.get_speed(veh_id) / self.max_speed),
                (self.k.vehicle.get_lane(veh_id) / MAX_LANES), edge_num
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
            headway = np.asarray([1000] * num_lanes) / headway_scale
            tailway = np.asarray([1000] * num_lanes) / headway_scale
            vel_in_front = np.asarray([0] * num_lanes) / self.max_speed
            vel_behind = np.asarray([0] * num_lanes) / self.max_speed

            lane_leaders = self.k.vehicle.get_lane_leaders(veh_id)
            lane_followers = self.k.vehicle.get_lane_followers(veh_id)
            lane_headways = self.k.vehicle.get_lane_headways(veh_id)
            lane_tailways = self.k.vehicle.get_lane_tailways(veh_id)
            headway[0:len(lane_headways)] = (
                np.asarray(lane_headways) / headway_scale)
            tailway[0:len(lane_tailways)] = (
                np.asarray(lane_tailways) / headway_scale)
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (
                        self.k.vehicle.get_speed(lane_leader) / self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = (self.k.vehicle.get_speed(lane_follower) /
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
        for edge in self.k.network.get_edge_list():
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(veh_ids) > 0:
                avg_speed = (sum(self.k.vehicle.get_speed(veh_ids)) /
                             len(veh_ids)) / self.max_speed
                density = len(veh_ids) / self.k.network.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        return np.concatenate((rl_obs, relative_obs, edge_obs))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        num_rl = self.k.vehicle.num_rl_vehicles
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return (rewards.desired_velocity(self) + rewards.rl_forward_progress(
            self, gain=0.1) - rewards.boolean_action_penalty(
                lane_change_acts, gain=1.0))

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.num_rl
        ub = [max_accel, 1] * self.num_rl

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def _apply_rl_actions(self, actions):
        """
        See parent class.

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands
        for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied,
        and sufficient time has passed, issue an acceleration like normal.
        """
        num_rl = self.k.vehicle.num_rl_vehicles
        acceleration = actions[::2][:num_rl]
        direction = np.round(actions[1::2])[:num_rl]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(),
                               key=self.k.vehicle.get_x_by_id)

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [
            self.time_counter <= self.env_params.additional_params[
                'lane_change_duration'] + self.k.vehicle.get_last_lc(veh_id)
            for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.

        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.
        """
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge='1',
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass


class BottleneckDesiredVelocityEnv(BottleneckEnv):
    """BottleneckDesiredVelocityEnv.

    Environment used to train vehicles to effectively pass through a
    bottleneck by specifying the velocity that RL vehicles should attempt to
    travel in certain regions of space.

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

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize BottleneckDesiredVelocityEnv."""
        super().__init__(env_params, sim_params, network, simulator)
        for p in ADDITIONAL_VSL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # default (edge, segment, controlled) status
        add_env_params = self.env_params.additional_params
        default = [(str(i), 1, True) for i in range(1, 6)]
        super(BottleneckDesiredVelocityEnv, self).__init__(env_params,
                                                           sim_params,
                                                           network)
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
            edge_length = self.k.network.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments + 1)

        # get info for observed segments
        self.obs_segments = additional_params.get("observed_segments", [])

        # number of segments for each edge
        self.num_obs_segments = [segment[1] for segment in self.obs_segments]

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.obs_slices = {}
        for edge, num_segments in self.obs_segments:
            edge_length = self.k.network.edge_length(edge)
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
                num_lanes = self.k.network.num_lanes(edge)
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
                    num_lanes = self.k.network.num_lanes(edge)
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
            num_obs += 4 * segment[1] * self.k.network.num_lanes(segment[0])
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
                    num_lanes = self.k.network.num_lanes(segment[0])
                    action_size += num_lanes * segment[1]
        add_params = self.env_params.additional_params
        max_accel = add_params.get("max_accel")
        max_decel = add_params.get("max_decel")
        return Box(
            low=-max_decel*self.sim_step, high=max_accel*self.sim_step,
            shape=(int(action_size), ), dtype=np.float32)

    def get_state(self):
        """Return aggregate statistics of different segments of the bottleneck.

        The state space of the system is defined by splitting the bottleneck up
        into edges and then segments in each edge. The class variable
        self.num_obs_segments specifies how many segments each edge is cut up
        into. Each lane defines a unique segment: we refer to this as a
        lane-segment. For example, if edge 1 has four lanes and three segments,
        then we have a total of 12 lane-segments. We will track the aggregate
        statistics of the vehicles in each lane segment.

        For each lane-segment we return the:

        * Number of vehicles on that segment.
        * Number of AVs (referred to here as rl_vehicles) in the segment.
        * The average speed of the vehicles in that segment.
        * The average speed of the rl vehicles in that segment.

        Finally, we also append the total outflow of the bottleneck over the
        last 20 * self.sim_step seconds.
        """
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.k.network.num_lanes(edge)
            num_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            num_rl_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            rl_vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            ids = self.k.vehicle.get_ids_by_edge(edge)
            lane_list = self.k.vehicle.get_lane(ids)
            pos_list = self.k.vehicle.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.obs_slices[edge],
                                          pos_list[i]) - 1
                if id in self.k.vehicle.get_rl_ids():
                    rl_vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_vehicles[segment, lane_list[i]] += 1

            # normalize

            num_vehicles /= NUM_VEHICLE_NORM
            num_rl_vehicles /= NUM_VEHICLE_NORM
            num_vehicles_list += num_vehicles.flatten().tolist()
            num_rl_vehicles_list += num_rl_vehicles.flatten().tolist()
            vehicle_speeds_list += vehicle_speeds.flatten().tolist()
            rl_speeds_list += rl_vehicle_speeds.flatten().tolist()

        unnorm_veh_list = np.asarray(num_vehicles_list) * NUM_VEHICLE_NORM
        unnorm_rl_list = np.asarray(num_rl_vehicles_list) * NUM_VEHICLE_NORM

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
            self.k.vehicle.get_outflow_rate(20 * self.sim_step) / 2000.0)
        return np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow]))

    def _apply_rl_actions(self, rl_actions):
        """
        RL actions are split up into 3 levels.

        * First, they're split into edge actions.
        * Then they're split into segment actions.
        * Then they're split into lane actions.
        """
        for rl_id in self.k.vehicle.get_rl_ids():
            edge = self.k.vehicle.get_edge(rl_id)
            lane = self.k.vehicle.get_lane(rl_id)
            if edge:
                # If in outer lanes, on a controlled edge, in a controlled lane
                if edge[0] != ':' and edge in self.controlled_edges:
                    pos = self.k.vehicle.get_position(rl_id)

                    if not self.symmetric:
                        num_lanes = self.k.network.num_lanes(edge)
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[int(lane) + bucket * num_lanes +
                                            self.action_index[edge]]
                    else:
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[bucket + self.action_index[edge]]

                    max_speed_curr = self.k.vehicle.get_max_speed(rl_id)
                    next_max = np.clip(max_speed_curr + action, 0.01, 23.0)
                    self.k.vehicle.set_max_speed(rl_id, next_max)

                else:
                    # set the desired velocity of the controller to the default
                    self.k.vehicle.set_max_speed(rl_id, 23.0)

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        if self.env_params.evaluate:
            if self.time_counter == self.env_params.horizon:
                reward = self.k.vehicle.get_outflow_rate(500)
            else:
                return 0
        else:
            reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / \
                (2000.0 * self.scaling)
        return reward

    def reset(self):
        """Reset the environment with a new inflow rate.

        The diverse set of inflows are used to generate a policy that is more
        robust with respect to the inflow rate. The inflow rate is update by
        creating a new network similar to the previous one, but with a new
        Inflow object with a rate within the additional environment parameter
        "inflow_range", which is a list consisting of the smallest and largest
        allowable inflow rates.

        **WARNING**: The inflows assume there are vehicles of type
        "followerstopper" and "human" within the VehicleParams object.
        """
        add_params = self.env_params.additional_params
        if add_params.get("reset_inflow"):
            inflow_range = add_params.get("inflow_range")
            flow_rate = np.random.uniform(
                min(inflow_range), max(inflow_range)) * self.scaling

            # We try this for 100 trials in case unexpected errors during
            # instantiation.
            for _ in range(100):
                try:
                    # introduce new inflows within the pre-defined inflow range
                    inflow = InFlows()
                    inflow.add(
                        veh_type="followerstopper",  # FIXME: make generic
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

                    # all other network parameters should match the previous
                    # environment (we only want to change the inflow)
                    additional_net_params = {
                        "scaling": self.scaling,
                        "speed_limit": self.net_params.
                        additional_params['speed_limit']
                    }
                    net_params = NetParams(
                        inflows=inflow,
                        additional_params=additional_net_params)

                    vehicles = VehicleParams()
                    vehicles.add(
                        veh_id="human",  # FIXME: make generic
                        car_following_params=SumoCarFollowingParams(
                            speed_mode=9,
                        ),
                        lane_change_controller=(SimLaneChangeController, {}),
                        routing_controller=(ContinuousRouter, {}),
                        lane_change_params=SumoLaneChangeParams(
                            lane_change_mode=0,  # 1621,#0b100000101,
                        ),
                        num_vehicles=1 * self.scaling)
                    vehicles.add(
                        veh_id="followerstopper",
                        acceleration_controller=(RLController, {}),
                        lane_change_controller=(SimLaneChangeController, {}),
                        routing_controller=(ContinuousRouter, {}),
                        car_following_params=SumoCarFollowingParams(
                            speed_mode=9,
                        ),
                        lane_change_params=SumoLaneChangeParams(
                            lane_change_mode=0,
                        ),
                        num_vehicles=1 * self.scaling)

                    # recreate the network object
                    self.network = self.network.__class__(
                        name=self.network.orig_name,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.initial_config,
                        traffic_lights=self.network.traffic_lights)
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
