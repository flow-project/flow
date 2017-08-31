import sumolib
from cistar.core.util import ensure_dir

class SumoParams():
    def __init__(self, port=None, time_step=0.1, vehicle_arrangement_shuffle=False, starting_position_shuffle=False,
                 emission_path="./data/", rl_speed_mode='no_collide', human_speed_mode='no_collide',
                 rl_lane_change_mode="no_lat_collide" , human_lane_change_mode ="no_lat_collide" ):
        """
        Configuration for running SUMO

        Specify the rl_speed_mode and human_speed_mode as the SUMO-defined speed
        mode used to constrain acceleration actions.
        The available speed modes are as follows:
         - "no_collide" (default): Human and RL cars are preventing from reaching speeds that may cause
                        crashes (also serves as a failsafe).
         - "aggressive": Human and RL cars are not limited by sumo with regard to their accelerations,
                         and can crash longitudinally

        Specify the SUMO-defined lane-changing mode used to constrain lane-changing actions
        The available lane-changing modes are as follows:
         - default: Human and RL cars can only safely change into lanes
         - "strategic": Human cars make lane changes in accordance with SUMO to provide speed boosts
         - "no_lat_collide": RL cars can lane change into any space, no matter how likely it is to crash
         - "aggressive": RL cars are not limited by sumo with regard to their lane-change actions,
                         and can crash longitudinally


        :param port: (Optional) Port for Traci to connect to; finds an empty port by default
        :param time_step: (Optional) seconds per simulation step; 0.1 by default
        :param vehicle_arrangement_shuffle: (Optional) determines if initial conditions of vehicles are shuffled
                                            at reset; False by default
        :param starting_position_shuffle: (Optional) determines if starting position of vehicles should be updated
                                          between rollouts; False by default
        :param emission_path: (Optional) Path to the folder in which to create the emissions output. Emissions output
                                          is not generated if this value is not specified
        :param rl_speed_mode:  (Optional) 'aggressive' or 'no collide'
        :param human_speed_mode: (Optional) 'aggressive' or 'no collide'
        :param rl_lane_change_mode:  (Optional) 'no_lat_collide' or 'strategic' or 'aggressive'
        :param human_lane_change_mode: (Optional) 'no_lat_collide' or 'strategic' or 'aggressive'
        """
        if not port:
            self.port = sumolib.miscutils.getFreeSocketPort()
        else:
            self.port = port
        self.time_step = time_step
        self.vehicle_arrangement_shuffle = vehicle_arrangement_shuffle
        self.starting_position_shuffle = starting_position_shuffle
        self.emission_path = emission_path
        self.rl_speed_mode = rl_speed_mode
        self.human_speed_mode = human_speed_mode
        self.rl_lane_change_mode = rl_lane_change_mode
        self.human_lane_change_mode = human_lane_change_mode

class EnvParams():
    def __init__(self, longitudinal_fail_safe='None', observation_vel_std=0, observation_pos_std=0, human_acc_std=0, rl_acc_std=0,
                 max_speed=55.0, lane_change_duration=None, shared_reward=False, shared_policy=False, additional_params = None):
        """

        :param longitudinal_fail_safe: Failsafe strategy to prevent bumper to bumper collisions
        :param observation_vel_std: observation (sensor) noise associated with velocity data
        :param observation_pos_std: observation (sensor) noise associated with position data
        :param human_acc_std: action (actuator) noise associated with human-driven vehicle acceleration
        :param rl_acc_std: action (actuator) noise associated with autonomous vehicle acceleration
        :param max_speed: max speed of vehicles in the simulation; defaults to 55 m/s
        :param lane_change_duration: ; defaults to 5 seconds
                                     lane changing duration is always present in the environment, but only used by
                                      sub-classes that apply lane changing
        :param shared_reward: (Boolean) use a shared reward; defaults to False
        :param shared_policy: (Boolean) use a shared policy; defaults to False
        :param additional_params: Specify additional environment params for a specific environment configuration
        """
        self.fail_safe = longitudinal_fail_safe
        self.observation_vel_std = observation_vel_std
        self.observation_pos_std = observation_pos_std
        self.human_acc_std = human_acc_std
        self.rl_acc_std = rl_acc_std
        self.max_speed = max_speed
        self.lane_change_duration = lane_change_duration
        self.shared_reward = shared_reward
        self.shared_policy = shared_policy
        self.additional_params = additional_params

    def get_additional_param(self, key):
        return self.additional_params[key]

    def get_lane_change_duration(self, time_step):
        if not self.lane_change_duration:
            return 5 / time_step
        else:
            return self.lane_change_duration / time_step