"""Base environment for the Bay Bridge."""
import numpy as np
from collections import defaultdict

from flow.envs import Env

EDGE_LIST = [
    '11198593', '236348360#1', '157598960', '11415208', '236348361',
    '11198599', '35536683', '11198595.0', '11198595.656.0', "gneE5",
    '340686911#3', '23874736', '119057701', '517934789', '236348364',
    '124952171', "gneE0", "11198599", "124952182.0", '236348360#0',
    '497579295', '340686911#2.0', '340686911#1', '394443191', '322962944',
    "32661309#1.0", "90077193#1.777", "90077193#1.0", "90077193#1.812",
    "gneE1", "183343422", "393649534", "32661316", "4757680", "124952179",
    "11189946", "119058993", "28413679", "11197898", "123741311", "123741303",
    "90077193#0", "28413687#0", "28413687#1", "11197889", "123741382#0",
    "123741382#1", "gneE3", "340686911#0.54.0", "340686911#0.54.54.0",
    "340686911#0.54.54.127.0", "340686911#2.35"
]

MAX_LANES = 24
NUM_EDGES = len(EDGE_LIST)
OBS_SPACE = 4 + 2 * NUM_EDGES + 4 * MAX_LANES
NUM_TRAFFIC_LIGHTS = 14

# number of vehicles a traffic light can observe in each lane
NUM_OBSERVED = 10
EDGE_BEFORE_TOLL = "gneE3"
TB_TL_ID = "gneJ4"
EDGE_AFTER_TOLL = "340686911#0.54.0"
NUM_TOLL_LANES = 20
TOLL_BOOTH_AREA = 100

EDGE_BEFORE_RAMP_METER = "340686911#0.54.54.0"
EDGE_AFTER_RAMP_METER = "340686911#0.54.54.127.0"
NUM_RAMP_METERS = 14
RAMP_METER_AREA = 80

MEAN_SECONDS_WAIT_AT_FAST_TRACK = 3
MEAN_SECONDS_WAIT_AT_TOLL = 15
FAST_TRACK_ON = range(6, 11)


class BayBridgeEnv(Env):
    """Base environment class for Bay Bridge scenarios.

    This class is responsible for mimicking the effects of the

    States
        No observations are issued by this class (i.e. empty list).

    Actions
        No actions are issued by this class.

    Rewards
        The reward is the average speed of vehicles in the network
        (temporarily).

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)
        self.edge_dict = defaultdict(list)
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES))
        self.tl_state = ""
        self.disable_tb = False
        self.disable_ramp_metering = False

        if "disable_tb" in env_params.additional_params:
            self.disable_tb = env_params.get_additional_param("disable_tb")

        if "disable_ramp_metering" in env_params.additional_params:
            self.disable_ramp_metering = env_params.get_additional_param(
                "disable_ramp_metering")

    def additional_command(self):
        """See parent class.

        This methods add traffic light and ramp metering control to the
        environment.
        """
        super().additional_command()
        # build a list of vehicles and their edges and positions
        self.edge_dict = defaultdict(list)
        # update the dict with all the edges in edge_list so we can look
        # forward for edges
        self.edge_dict.update(
            (k, [[] for _ in range(MAX_LANES)]) for k in EDGE_LIST)
        for veh_id in self.k.vehicle.get_ids():
            edge = self.k.vehicle.get_edge(veh_id)
            if edge not in self.edge_dict:
                self.edge_dict.update({edge: [[] for _ in range(MAX_LANES)]})
            lane = self.k.vehicle.get_lane(veh_id)  # integer
            pos = self.k.vehicle.get_position(veh_id)

            # perform necessary lane change actions to keep vehicle in the
            # right route
            self.edge_dict[edge][lane].append((veh_id, pos))
            if edge == "124952171" and lane == 1:
                self.k.vehicle.apply_lane_change([veh_id], direction=[1])

        if not self.disable_tb:
            self.apply_toll_bridge_control()
        if not self.disable_ramp_metering:
            self.ramp_meter_lane_change_control()

    def ramp_meter_lane_change_control(self):
        """Control the lane changing behavior.

        Specify/Toggle the lane changing behavior of the vehicles depending on
        factors like whether or not they are before the toll.
        """
        cars_that_have_left = []
        for veh_id in self.cars_before_ramp:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                if self.simulator == 'traci':
                    lane_change_mode = self.cars_before_ramp[veh_id][
                        'lane_change_mode']
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                color = self.cars_before_ramp[veh_id]['color']
                self.k.vehicle.set_color(veh_id, color)

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            self.cars_before_ramp.__delitem__(veh_id)

        for lane in range(NUM_RAMP_METERS):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for car in cars_in_lane:
                veh_id, pos = car
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        if self.simulator == 'traci':
                            # Disable lane changes inside Toll Area
                            lane_change_mode = self.k.kernel_api.vehicle.\
                                getLaneChangeMode(veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (0, 255, 255))
                        self.cars_before_ramp[veh_id] = {
                            "lane_change_mode": lane_change_mode,
                            "color": color
                        }

    def apply_toll_bridge_control(self):
        """Apply control to the toll bridge."""
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.k.vehicle.get_lane(veh_id)
                if self.simulator == 'traci':
                    lane_change_mode = \
                        self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.k.vehicle.set_color(veh_id, color)
                if lane not in FAST_TRACK_ON:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            loc=MEAN_SECONDS_WAIT_AT_TOLL / self.sim_step,
                            scale=1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            loc=MEAN_SECONDS_WAIT_AT_FAST_TRACK /
                            self.sim_step,
                            scale=1 / self.sim_step))

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
                        if self.simulator == 'traci':
                            # Disable lane changes inside Toll Area
                            lc_mode = self.k.kernel_api.vehicle.\
                                getLaneChangeMode(veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lc_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (255, 0, 255))
                        self.cars_waiting_for_toll[veh_id] = {
                            "lane_change_mode": lc_mode,
                            "color": color
                        }
                    else:
                        if pos > 120:
                            if self.toll_wait_time[lane] < 0:
                                traffic_light_states[lane] = "G"
                            else:
                                traffic_light_states[lane] = "r"
                                self.toll_wait_time[lane] -= 1

        new_tls_state = "".join(traffic_light_states)

        if new_tls_state != self.tl_state:
            self.tl_state = new_tls_state
            self.k.traffic_light.set_state(
                node_id=TB_TL_ID, state=new_tls_state)

    # TODO: decide on a good reward function
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

    ###########################################################################
    #         The below methods need to be updated by child classes.          #
    ###########################################################################

    def _apply_rl_actions(self, rl_actions):
        """See parent class.

        To be implemented by child classes.
        """
        pass

    def get_state(self):
        """See parent class.

        To be implemented by child classes.
        """
        return []
