
from flow.envs.loop_accel import SimpleAccelerationEnvironment

import numpy as np
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple


class ShepherdingEnv(SimpleAccelerationEnvironment):

    def compute_reward(self, state, rl_actions, **kwargs):
        # num_non_rl = (self.vehicles.num_vehicles-self.vehicles.num_rl_vehicles)
        # desired_vel = np.array([self.env_params.additional_params["target_velocity"]] * num_non_rl)
        # maxdiff = np.linalg.norm(np.array([0] * num_non_rl) - desired_vel)
        # curr_vel = [np.array(self.vehicles.get_speed(veh_id)) for veh_id in self.vehicles.get_sumo_ids()]
        #
        # max_diff_vel = np.max(np.abs(desired_vel - curr_vel))
        # min_diff_vel = np.min(np.abs(desired_vel - curr_vel))
        # norm_diff_vel = np.linalg.norm(np.abs(desired_vel - curr_vel))
        # # accel = self.vehicles.get_accel(veh_id="all")
        # # deaccel =  np.linalg.norm([min(0, x) for x in accel])
        # # print(max(curr_vel), min(curr_vel), self.env_params.additional_params["target_velocity"] - diff_vel)
        #
        # rl_speeds = np.linalg.norm([max(10 - x,0) for x in self.vehicles.get_speed(self.vehicles.get_rl_ids())])
        # print((self.env_params.additional_params["target_velocity"] - deviation)**2)

        deviation = np.abs(self.env_params.additional_params["target_velocity"]-self.vehicles.get_speed("aggressive-human_0"))
        return (self.env_params.additional_params["target_velocity"] - deviation)**2

    @property
    def action_space(self):
        """
        See parent class

        Actions are:
         - a (continuous) acceleration from max-deacc to max-acc
         - a (continuous) lane-change action from -1 to 1, used to determine the
           lateral direction the vehicle will take.
        """
        max_deacc = self.env_params.max_deacc
        max_acc = self.env_params.max_acc

        lb = [-abs(max_deacc), -1] * self.vehicles.num_rl_vehicles
        ub = [max_acc, 1] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub))

    @property
    def observation_space(self):
        """
        See parent class

        An observation consists of the velocity, absolute position, and lane
        index of each vehicle in the fleet
        """
        speed = Box(low=-np.inf, high=1, shape=(self.vehicles.num_vehicles,))
        lane = Box(low=0, high=self.scenario.lanes - 1, shape=(self.vehicles.num_vehicles,))
        absolute_pos = Box(low=0., high=1, shape=(self.vehicles.num_vehicles,))
        return Tuple((speed, absolute_pos, lane))

    def get_state(self):
        """
        See parent class

        The state is an array the velocities, absolute positions, and lane
        numbers for each vehicle.
        """
        scaled_pos = [self.vehicles.get_absolute_position(veh_id) /
                      self.scenario.length for veh_id in self.sorted_ids]
        scaled_vel = [self.vehicles.get_speed(veh_id) /
                      self.env_params.get_additional_param("target_velocity")
                      for veh_id in self.sorted_ids]
        lane = [self.vehicles.get_speed(veh_id) for veh_id in self.sorted_ids]
        return np.array([[scaled_vel[i], scaled_pos[i], lane[i]]
                         for i in range(len(self.sorted_ids))])


    def apply_rl_actions(self, actions):
        """
        See parent class

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands for the duration of the lane
        change and return negative rewards for actions during that lane change.
        if a lane change isn't applied, and sufficient time has passed, issue an
        acceleration like normal.
        """
        acceleration = actions[::2]
        direction = np.round(actions[1::2])

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.rl_ids]
        # sorted_rl_ids = self.rl_ids

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.timer <= self.lane_change_duration + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = np.array([0] * sum(non_lane_changing_veh))

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)


    def __init__(self, env_params, sumo_params, scenario):
        super().__init__(env_params, sumo_params, scenario)
        self.tau_delay_aggro_driver = False
        for veh_type, params in self.vehicles.types:
            if veh_type == "aggressive-human":
                self.tau_delay_aggro_driver = True
                self.impatient_tau = params["tau"]
                self.init_tau = params["tau"]

                self.time_steps_stuck = 0

    def additional_command(self):
        # print(self.vehicles.get_speed("aggressive-human_0"))
        if self.tau_delay_aggro_driver:
            if self.vehicles.get_speed("aggressive-human_0") < 20 and self.vehicles.get_headway("aggressive-human_0") < 10:
                self.time_steps_stuck += 1
            else:
                self.time_steps_stuck = 0
                if self.impatient_tau != self.init_tau:
                    self.impatient_tau = self.init_tau
                    self.traci_connection.vehicle.setTau("aggressive-human_0", self.impatient_tau)

            if self.time_steps_stuck > 50:
                if self.impatient_tau < 3.0:
                    self.impatient_tau *= 1.05
                    self.traci_connection.vehicle.setTau("aggressive-human_0", self.impatient_tau)

    def sort_by_position(self):
        """
        Sorts the vehicle ids first by type and then by position.
        The RL cars are first by default.
        Returns
        -------
        sorted_ids: list
            a list of all vehicle IDs sorted by type andt then by position
        sorted_extra_data: list or tuple
            an extra component (list, tuple, etc...) containing extra sorted
            data, such as positions. If no extra component is needed, a value
            of None should be returned
        """
        non_aggressive_humans = list(self.vehicles.get_sumo_ids())
        non_aggressive_humans.remove("aggressive-human_0")
        sorted_indx = np.argsort(self.vehicles.get_absolute_position(non_aggressive_humans))
        sorted_ids = np.concatenate([np.array(["rl_0", "rl_1", "rl_2", "aggressive-human_0"]), np.array(non_aggressive_humans)[sorted_indx]])
        return sorted_ids, None