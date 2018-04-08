import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.base_env import Env


class GreenWaveEnv(Env):
    """Environment used to train traffic lights to regulate traffic flow
    through an n x m grid.

    States
    ------
    An observation is the distance of each vehicle to its intersection, a
    number uniquely identifying which edge the vehicle is on, and the speed of
    the vehicle.

    Actions
    -------
    The action space consist of a list of float variables ranging from 0-1
    specifying whether a traffic light is supposed to switch or not. The
    actions are sent to the traffic light in the grid from left to right and
    then top to bottom.

    Rewards
    -------
    The reward is the two-norm of the difference between the speed of all
    vehicles in the network and some desired speed.

    Termination
    -----------
    A rollout is terminated once the time horizon is reached.

    Additional
    ----------
    Vehicles are rerouted to the start of their original routes once they reach
    the end of the network in order to ensure a constant number of vehicles.
    """
    def __init__(self, env_params, sumo_params, scenario):
        self.grid_array = scenario.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        self.num_traffic_lights = self.rows * self.cols

        super().__init__(env_params, sumo_params, scenario)

        # Saving env variables for plotting
        self.steps = env_params.additional_params.get('num_steps', 0)
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'velocities': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'positions': np.zeros((self.steps, self.vehicles.num_vehicles))}
        self.node_mapping = scenario.get_node_mapping()

        # keeps track of the last time the light was allowed to change.
        self.last_change = np.zeros((self.rows * self.cols, 3))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]
        for i in range(self.rows * self.cols):
            self.traci_connection.trafficlights.setRedYellowGreenState(
                'center' + str(i), "GGGrrrGGGrrr")
            self.last_change[i, 2] = 1

        # Additional Information for Plotting
        self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        for i, veh_id in enumerate(self.vehicles.get_ids()):
            edge = self.vehicles.get_edge(veh_id)
            for key in self.edge_mapping:
                if key in edge:
                    self.edge_mapping[key].append(i)
                    break

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(self.num_traffic_lights,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(low=0, high=1, shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        dist_to_intersec = Box(low=0., high=np.inf,
                               shape=(self.vehicles.num_vehicles,),
                               dtype=np.float32)
        edge_num = Box(low=0., high=1, shape=(self.vehicles.num_vehicles,),
                       dtype=np.float32)
        traffic_lights = Box(low=0., high=np.inf,
                             shape=(3 * self.rows * self.cols),
                             dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        # compute the normalizers
        max_speed = self.max_speed
        max_dist = max(self.scenario.short_length,
                       self.scenario.long_length,
                       self.scenario.inner_length)

        # get the state arrays
        speeds = [self.vehicles.get_speed(veh_id) / max_speed for veh_id in
                  self.vehicles.get_ids()]
        dist_to_intersec = [self.get_distance_to_intersection(veh_id)/max_dist
                            for veh_id in self.vehicles.get_ids()]
        edges = [self._convert_edge(self.vehicles.get_edge(veh_id)) / (
            self.scenario.num_edges - 1) for veh_id in self.vehicles.get_ids()]

        state = [speeds, dist_to_intersec, edges,
                 self.last_change.flatten().tolist()]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        # convert values less than 0.5 to zero and above to 1. 0's indicate
        # that should not switch the direction
        rl_mask = rl_actions > 0.5

        for i, action in enumerate(rl_actions):
            # check if our timer has exceeded the yellow phase, meaning it
            # should switch to red
            if self.last_change[i, 2] == 0:  # currently yellow
                self.last_change[i, 0] += self.sim_step
                if self.last_change[i, 0] >= self.min_switch_time:
                    if self.last_change[i, 1] == 0:
                        self.traffic_lights.set_state(
                            node_id='center{}'.format(i),
                            state="GGGrrrGGGrrr", env=self)
                    else:
                        self.traffic_lights.set_state(
                            node_id='center{}'.format(i),
                            state='rrrGGGrrrGGG', env=self)
                    self.last_change[i, 2] = 1
            else:
                if rl_mask[i]:
                    if self.last_change[i, 1] == 0:
                        self.traffic_lights.set_state(
                            node_id='center{}'.format(i),
                            state='yyyrrryyyrrr', env=self)
                    else:
                        self.traffic_lights.set_state(
                            node_id='center{}'.format(i),
                            state='rrryyyrrryyy', env=self)
                    self.last_change[i, 0] = 0.0
                    self.last_change[i, 1] = not self.last_change[i, 1]
                    self.last_change[i, 2] = 0

    def compute_reward(self, state, rl_actions, **kwargs):
        return rewards.penalize_tl_changes(self, rl_actions >= 0.5, gain=1)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determines the smallest distance from the current vehicle's position
        to any of the intersections.

        Parameters
        ----------
        veh_ids: str
            vehicle identifier

        Yields
        ------
        tup
            1st element: distance to closest intersection
            2nd element: intersection ID (which also specifies which side of
            the intersection the vehicle will be arriving at)
        """
        if isinstance(veh_ids, list):
            return [self.find_intersection_dist(veh_id) for veh_id in veh_ids]
        else:
            return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from the vehicle's current position to the position
        of the node it is heading toward."""
        edge_id = self.vehicles.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.scenario.edge_length(edge_id)
        relative_pos = self.vehicles.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        """Converts the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bot.

        The values are are zero indexed.

        Parameters
        ----------
        edges: list <str> or str
            name of the edge(s)

        Yields
        ------
        list <int> or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Utility function for convert_edge"""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols+1) * self.rows * 2) \
                    + ((self.rows+1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def additional_command(self):
        """Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge."""
        for veh_id in self.vehicles.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Checks if an edge is the final edge. If it is return the route it
        should start off at."""
        edge = self.vehicles.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going tyo remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            route_id = "route" + route_id
            # remove the vehicle
            self.traci_connection.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            type_id = self.vehicles.get_state(veh_id, "type")
            lane_index = self.vehicles.get_lane(veh_id)
            self.traci_connection.vehicle.addFull(
                veh_id, route_id, typeID=str(type_id),
                departLane=str(lane_index),
                departPos="0", departSpeed="max")

    def record_obs_var(self):
        """Records the edges and velocities in the observation variables."""
        for i, veh_id in enumerate(self.vehicles.get_ids()):
            self.obs_var_labels['velocities'][
                self.time_counter - 1, i] = self.vehicles.get_speed(veh_id)
            self.obs_var_labels['edges'][self.time_counter - 1, i] = \
                self._convert_edge(self.vehicles.get_edge(veh_id))
            x = self.get_x_by_id(veh_id)
            if x > 2000:  # hardcode
                x = 0
            self.obs_var_labels['positions'][self.time_counter - 1, i] = x


class GreenWaveTestEnv(GreenWaveEnv):
    """
    Class that overrides RL methods of green wave so we can test
    construction without needing to specify RL methods
    """
    def _apply_rl_actions(self, rl_actions):
        pass

    def compute_reward(self, state, rl_actions, **kwargs):
        return 0
