import numpy as np
import re

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.base_env import Env

from matplotlib import pyplot as plt
from flow.scenarios.subnetworks import *
from flow.envs.loop.loop_accel import AccelCNNEnv

ADDITIONAL_ENV_PARAMS = {

}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}

class MiniCityTrafficLightsEnv(Env):
    """Environment used to train traffic lights to regulate traffic flow
    through an n x m grid.

    Required from env_params:

    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL
      options are "controlled" and "actuated"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sumo_params, scenario):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.rows = 1
        self.cols = 1
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sumo_params, scenario)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'velocities': np.zeros((self.steps, self.vehicles.num_vehicles)),
            'positions': np.zeros((self.steps, self.vehicles.num_vehicles))
        }

        # keeps track of the last time the light was allowed to change.
        self.last_change = np.zeros((self.rows * self.cols, 3))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        # Additional Information for Plotting
        self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        for i, veh_id in enumerate(self.vehicles.get_ids()):
            edge = self.vehicles.get_edge(veh_id)
            for key in self.edge_mapping:
                if key in edge:
                    self.edge_mapping[key].append(i)
                    break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2 ** self.num_traffic_lights)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        height = 42
        width = 42
        return Box(0., 1., [height, width, 5])

    def get_state(self):
        return np.ones(shape=(5, 42, 42))

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0.5 to zero and above to 1. 0's indicate
            # that should not switch the direction
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            # check if our timer has exceeded the yellow phase, meaning it
            # should switch to red

            # keeps track of the last time the light was allowed to change.
            #self.last_change = np.zeros((self.rows * self.cols, 3))

            # when this hits min_switch_time we change from yellow to red
            # the second column indicates the direction that is currently being
            # allowed to flow. 0 is flowing top to bottom, 1 is left to right
            # For third column, 0 signifies yellow and 1 green or red
            if self.last_change[i, 2] == 0:  # currently yellow
                self.last_change[i, 0] += self.sim_step
                if self.last_change[i, 0] >= self.min_switch_time:
                    if self.last_change[i, 1] == 0:
                        self.traffic_lights.set_state(
                            node_id='n_i4',
                            state="GGGrrrGGGrrr",
                            env=self)
                    else:
                        self.traffic_lights.set_state(
                            node_id='n_i4',
                            state='rrrGGGrrrGGG',
                            env=self)
                    self.last_change[i, 2] = 1
            else:
                if action:
                    if self.last_change[i, 1] == 0:
                        self.traffic_lights.set_state(
                            node_id='n_i4',
                            state='rrryyyrrryyy',
                            env=self)
                    else:
                        self.traffic_lights.set_state(
                            node_id='n_i4',
                            state='yyyrrryyyrrr',
                            env=self)
                    self.last_change[i, 0] = 0.0
                    self.last_change[i, 1] = not self.last_change[i, 1]
                    self.last_change[i, 2] = 0

    # def compute_reward(self, rl_actions, **kwargs):
    #     """See class definition."""
    #     return rewards.penalize_tl_changes(rl_actions >= 0.5, gain=1.0)
    #     # reward = self.vehicles.get_outflow_rate(10 * self.sim_step) / \
    #     #          (2000.0 * 100)
    #     # return reward

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        max_speed = self.scenario.max_speed

        speed = self.vehicles.get_speed(self.vehicles.get_ids())
        if len(self.vehicles.get_ids()) ==0 :
            return 0
        else :
            return (0.8*np.mean(speed) - 0.2*np.std(speed))/max_speed


class AccelCNNSubnetEnv(AccelCNNEnv):

    # Currently has a bug with "sights_buffer / 255" in original AccelCNNEnv
    # Using cropped frame buffer as state instead
    def get_state(self, **kwargs):
        """See class definition."""
        cropped_frame_buffer = np.squeeze(np.array(self.frame_buffer))
        cropped_frame_buffer = np.moveaxis(cropped_frame_buffer, 0, -1).T
        return cropped_frame_buffer / 255.

    def render(self, reset=False, buffer_length=5):
        """Render a frame.
        Parameters
        ----------
        reset: bool
            set to True to reset the buffer
        buffer_length: int
            length of the buffer
        """
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]

                # Crop self.frame_buffer to subnetwork only
                for frame in self.frame_buffer:
                    subnet_xmin = self.env_params.additional_params['xmin']
                    subnet_xmax = self.env_params.additional_params['xmax']
                    subnet_ymin = self.env_params.additional_params['ymin']
                    subnet_ymax = self.env_params.additional_params['ymax']
                    frame = frame[subnet_ymin:subnet_ymax,
                                  subnet_xmin:subnet_xmax, :]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    next_frame = self.frame.copy()
                    subnet_xmin = self.env_params.additional_params['xmin']
                    subnet_xmax = self.env_params.additional_params['xmax']
                    subnet_ymin = self.env_params.additional_params['ymin']
                    subnet_ymax = self.env_params.additional_params['ymax']
                    next_frame = next_frame[subnet_ymin:subnet_ymax,
                                            subnet_xmin:subnet_xmax, :]

                    # Save a cropped image to current executing directory for debug
                    plt.imsave('test_subnet_crop.png', next_frame)

                    self.frame_buffer.append(next_frame)
                    self.sights_buffer.append(self.sights.copy())

                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)


class AccelCNNSubnetTrainingEnv(MiniCityTrafficLightsEnv):

    @property
    def observation_space(self):
        """See class definition."""
        subnet_spec = \
            SUBNET_CROP[self.env_params.additional_params['subnetwork']]
        subnet_xmin = self.env_params.additional_params['xmin']
        subnet_xmax = self.env_params.additional_params['xmax']
        subnet_ymin = self.env_params.additional_params['ymin']
        subnet_ymax = self.env_params.additional_params['ymax']
        height = subnet_ymax - subnet_ymin
        width = subnet_xmax - subnet_xmin
        channel = 6
        return Box(0., 1., [height, width, channel])

    # Currently has a bug with "sights_buffer / 255" in original AccelCNNEnv
    # Using cropped frame buffer as state instead
    def get_state(self, **kwargs):
        """See class definition."""
        cropped_frame_buffer = np.asarray(self.frame_buffer.copy())
        cropped_frame_buffer = np.dstack((cropped_frame_buffer[0,...],
                                          cropped_frame_buffer[1,...],))
        cropped_frame_buffer = cropped_frame_buffer.T   

        return cropped_frame_buffer / 255.

    def render(self, reset=False, buffer_length=5):
        """Render a frame.
        Parameters
        ----------
        reset: bool
            set to True to reset the buffer
        buffer_length: int
            length of the buffer
        """
        if self.sumo_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            subnet_spec = \
                SUBNET_CROP[self.env_params.additional_params['subnetwork']]
            subnet_xmin = self.env_params.additional_params['xmin']
            subnet_xmax = self.env_params.additional_params['xmax']
            subnet_ymin = self.env_params.additional_params['ymin']
            subnet_ymax = self.env_params.additional_params['ymax']
            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]

                # Crop self.frame_buffer to subnetwork only
                for idx, frame in enumerate(self.frame_buffer):
                    self.frame_buffer[idx] = frame[subnet_ymin:subnet_ymax, 
                                                  subnet_xmin:subnet_xmax, :]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    next_frame = self.frame.copy()
                    next_frame = next_frame[subnet_ymin:subnet_ymax,
                                            subnet_xmin:subnet_xmax, :]
                    self.frame_buffer.append(next_frame)
                    self.sights_buffer.append(self.sights.copy())
                    # Save a cropped image to current executing directory for debug
                    plt.imsave('test_subnet_crop{}-{}-{}-{}.png'.format(subnet_xmin,subnet_xmax,subnet_ymin,subnet_ymax), next_frame)
                    # Do this only when you are debugging (: It's slow.
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)
        #print("RENDER///////////////////////////////////////////////////////")
        #print("Frame buffer:")
        #for frame in self.frame_buffer:
        #    print(np.asarray(frame).shape)
        #print("Time:", self.time_counter)
        #print("///////////////////////////////////////////////////////RENDER")        

    # # ===============================
    # # ============ UTILS ============
    # # ===============================

    # def record_obs_var(self):
    #     """
    #     Records velocities and edges in self.obs_var_labels at each time
    #     step. This is used for plotting.
    #     """
    #     for i, veh_id in enumerate(self.vehicles.get_ids()):
    #         self.obs_var_labels['velocities'][
    #             self.time_counter - 1, i] = self.vehicles.get_speed(veh_id)
    #         self.obs_var_labels['edges'][self.time_counter - 1, i] = \
    #             self._convert_edge(self.vehicles.get_edge(veh_id))
    #         x = self.get_x_by_id(veh_id)
    #         if x > 2000:  # hardcode
    #             x = 0
    #         self.obs_var_labels['positions'][self.time_counter - 1, i] = x

    # def get_distance_to_intersection(self, veh_ids):
    #     """Determines the smallest distance from the current vehicle's position
    #     to any of the intersections.

    #     Parameters
    #     ----------
    #     veh_ids: str
    #         vehicle identifier

    #     Returns
    #     -------
    #     tup
    #         1st element: distance to closest intersection
    #         2nd element: intersection ID (which also specifies which side of
    #         the intersection the vehicle will be arriving at)
    #     """
    #     if isinstance(veh_ids, list):
    #         return [self.find_intersection_dist(veh_id) for veh_id in veh_ids]
    #     else:
    #         return self.find_intersection_dist(veh_ids)

    # def find_intersection_dist(self, veh_id):
    #     """Return distance from the vehicle's current position to the position
    #     of the node it is heading toward."""
    #     edge_id = self.vehicles.get_edge(veh_id)
    #     # FIXME this might not be the best way of handling this
    #     if edge_id == "":
    #         return -10
    #     if 'center' in edge_id:
    #         return 0
    #     edge_len = self.scenario.edge_length(edge_id)
    #     relative_pos = self.vehicles.get_position(veh_id)
    #     dist = edge_len - relative_pos
    #     return dist

    # def sort_by_intersection_dist(self):
    #     """Sorts the vehicle ids of vehicles in the network by their distance
    #     to the intersection.

    #     Returns
    #     -------
    #     sorted_ids: list
    #         a list of all vehicle IDs sorted by position
    #     sorted_extra_data: list or tuple
    #         an extra component (list, tuple, etc...) containing extra sorted
    #         data, such as positions. If no extra component is needed, a value
    #         of None should be returned
    #     """
    #     ids = self.vehicles.get_ids()
    #     sorted_indx = np.argsort(self.get_distance_to_intersection(ids))
    #     sorted_ids = np.array(ids)[sorted_indx]
    #     return sorted_ids

    # def _convert_edge(self, edges):
    #     """Converts the string edge to a number.

    #     Start at the bottom left vertical edge and going right and then up, so
    #     the bottom left vertical edge is zero, the right edge beside it  is 1.

    #     The numbers are assigned along the lowest column, then the lowest row,
    #     then the second lowest column, etc. Left goes before right, top goes
    #     before bot.

    #     The values are are zero indexed.

    #     Parameters
    #     ----------
    #     edges: list <str> or str
    #         name of the edge(s)

    #     Returns
    #     -------
    #     list <int> or int
    #         a number uniquely identifying each edge
    #     """
    #     if isinstance(edges, list):
    #         return [self._split_edge(edge) for edge in edges]
    #     else:
    #         return self._split_edge(edges)

    # def _split_edge(self, edge):
    #     """Utility function for convert_edge"""
    #     if edge:
    #         if edge[0] == ":":  # center
    #             center_index = int(edge.split("center")[1][0])
    #             base = ((self.cols + 1) * self.rows * 2) \
    #                 + ((self.rows + 1) * self.cols * 2)
    #             return base + center_index + 1
    #         else:
    #             pattern = re.compile(r"[a-zA-Z]+")
    #             edge_type = pattern.match(edge).group()
    #             edge = edge.split(edge_type)[1].split('_')
    #             row_index, col_index = [int(x) for x in edge]
    #             if edge_type in ['bot', 'top']:
    #                 rows_below = 2 * (self.cols + 1) * row_index
    #                 cols_below = 2 * (self.cols * (row_index + 1))
    #                 edge_num = rows_below + cols_below + 2 * col_index + 1
    #                 return edge_num if edge_type == 'bot' else edge_num + 1
    #             if edge_type in ['left', 'right']:
    #                 rows_below = 2 * (self.cols + 1) * row_index
    #                 cols_below = 2 * (self.cols * row_index)
    #                 edge_num = rows_below + cols_below + 2 * col_index + 1
    #                 return edge_num if edge_type == 'left' else edge_num + 1
    #     else:
    #         return 0

    # def additional_command(self):
    #     """Used to insert vehicles that are on the exit edge and place them
    #     back on their entrance edge."""
    #     for veh_id in self.vehicles.get_ids():
    #         self._reroute_if_final_edge(veh_id)

    # def _reroute_if_final_edge(self, veh_id):
    #     """Checks if an edge is the final edge. If it is return the route it
    #     should start off at."""
    #     edge = self.vehicles.get_edge(veh_id)
    #     if edge == "":
    #         return
    #     if edge[0] == ":":  # center edge
    #         return
    #     pattern = re.compile(r"[a-zA-Z]+")
    #     edge_type = pattern.match(edge).group()
    #     edge = edge.split(edge_type)[1].split('_')
    #     row_index, col_index = [int(x) for x in edge]

    #     # find the route that we're going to place the vehicle on if we are
    #     # going to remove it
    #     route_id = None
    #     if edge_type == 'bot' and col_index == self.cols:
    #         route_id = "bot{}_0".format(row_index)
    #     elif edge_type == 'top' and col_index == 0:
    #         route_id = "top{}_{}".format(row_index, self.cols)
    #     elif edge_type == 'left' and row_index == 0:
    #         route_id = "left{}_{}".format(self.rows, col_index)
    #     elif edge_type == 'right' and row_index == self.rows:
    #         route_id = "right0_{}".format(col_index)

    #     if route_id is not None:
    #         route_id = "route" + route_id
    #         # remove the vehicle
    #         self.traci_connection.vehicle.remove(veh_id)
    #         # reintroduce it at the start of the network
    #         type_id = self.vehicles.get_state(veh_id, "type")
    #         lane_index = self.vehicles.get_lane(veh_id)
    #         self.traci_connection.vehicle.addFull(
    #             veh_id,
    #             route_id,
    #             typeID=str(type_id),
    #             departLane=str(lane_index),
    #             departPos="0",
    #             departSpeed="max")
    #         speed_mode = self.vehicles.type_parameters[type_id]["speed_mode"]
    #         self.traci_connection.vehicle.setSpeedMode(veh_id, speed_mode)

    # def k_closest_to_intersection(self, edges, k):
    #     """
    #     Return the veh_id of the k closest vehicles to an intersection for
    #     each edge. Performs no check on whether or not edge is going toward an
    #     intersection or not. Does no padding
    #     """
    #     if k < 0:
    #         raise IndexError("k must be greater than 0")
    #     dists = []

    #     def sort_lambda(veh_id):
    #         return self.get_distance_to_intersection(veh_id)

    #     if isinstance(edges, list):
    #         for edge in edges:
    #             vehicles = self.vehicles.get_ids_by_edge(edge)
    #             dist = sorted(
    #                 vehicles,
    #                 key=sort_lambda
    #             )
    #             dists += dist[:k]
    #     else:
    #         vehicles = self.vehicles.get_ids_by_edge(edges)
    #         dist = sorted(
    #             vehicles,
    #             key=lambda veh_id: self.get_distance_to_intersection(veh_id))
    #         dists += dist[:k]
    #     return dists
