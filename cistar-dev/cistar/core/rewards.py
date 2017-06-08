import numpy as np


class Reward:
    """
    class containing methods for computing rl rewards.

    The rewards consist of two components:
     - reward function: The defining function used to train vehicles to perform certain tasks.
     - non compliance: the non-compliance component penalizes vehicles for performing tasks
       not requested by rl vehicles (not lane changing when they were requested to, or
       performing an acceleration that deviates from the requested value)
    """

    def __init__(self, reward_type, compute_non_compliance=False, non_compliance_params={},
                 teach_non_collision=True, num_vehicles=None, num_rl_vehicles=None):
        """
        Instantiates a rewards class. This is used to specify the type of non_compliance
        method the reward function would like to use,

        :param compute_non_compliance:
        :param non_compliance_params:
        :param teach_non_collision:
        :param num_vehicles: number of vehicles in the system. If no number is specified,
        the value is implicitly determined from the observation space
        :param num_rl_vehicles: number of rl vehicles in the system (only needed if
        non_compliance is being computed); If no number is specified, the value is
        implicitly determined from the non_compliance space
        """
        self.reward_type = reward_type

        self.compute_non_compliance = compute_non_compliance

        self.non_compliance_params = {}

        if self.compute_non_compliance:
            if "acceleration_type" in non_compliance_params:
                self.non_compliance_params["acceleration_type"] = non_compliance_params["acceleration_type"]

                if self.non_compliance_params["acceleration_type"] == "function":
                    self.non_compliance_params["acceleration_function"] = non_compliance_params["acceleration_function"]
            else:
                self.non_compliance_params["acceleration_type"] = "all_or_nothing"

            if "acceleration_gain" in non_compliance_params:
                self.non_compliance_params["acceleration_gain"] = non_compliance_params["acceleration_gain"]
            else:
                self.non_compliance_params["acceleration_gain"] = 1

            if "lane_change_type" in non_compliance_params:
                self.non_compliance_params["lane_change_type"] = non_compliance_params["lane_change_type"]

                if self.non_compliance_params["lane_change_type"] == "function":
                    self.non_compliance_params["lane_change_function"] = non_compliance_params["lane_change_function"]
            else:
                self.non_compliance_params["lane_change_type"] = "all_or_nothing"

            if "lane_change_gain" in non_compliance_params:
                self.non_compliance_params["lane_change_gain"] = non_compliance_params["lane_change_gain"]
            else:
                self.non_compliance_params["lane_change_gain"] = 1

        self.teach_non_collision = teach_non_collision

        self.num_vehicles = num_vehicles

        self.num_rl_vehicles = num_rl_vehicles

    def make_reward_positive(self, max_reward_penalty, max_lane_change_penalty, max_acceleration_penalty):
        """
        This function is used to teach vehicles not to crash by ensuring that all
        rewards are positive. As a result, if vehicles crash and end a rollout
        prematurely, the early end of the rollout serves as a penalty to the reward
        in itself.

        :param max_penalty: the maximum penalty the system can incur in any single
        dimension
        :return: A value symbolic of the magnitude of the maximum penalty over the
        entire vector space, which, once added to reward function, makes it positive.
        """
        reward_magnitude = np.linalg.norm([max_reward_penalty] * self.num_vehicles)

        # TODO: fix how non_compliance is set
        if self.compute_non_compliance:
            lane_change_magnitude = max_lane_change_penalty * self.num_rl_vehicles
            acceleration_magnitude = max_acceleration_penalty * self.num_rl_vehicles
        else:
            lane_change_magnitude = 0
            acceleration_magnitude = 0

        return reward_magnitude + lane_change_magnitude + acceleration_magnitude

    def compute_reward(self, state, rl_actions, **kwargs):
        """
        Primary function for computing the reward function. Includes making the reward
        positive in order to teach vehicles not to crash, and applying the specified method
        of penalizing non_compliance.

        :param state:
        :param rl_actions:
        :param kwargs:
        :return:
        """
        reward = 0

        if self.num_vehicles is None:
            self.num_vehicles = state.shape[0]  # check if this is supposed to be 0 or 1

        if self.reward_type == "target_velocity":
            reward = self.distance_from_target_velocity(state, rl_actions, **kwargs)

        if self.compute_non_compliance:
            if self.num_rl_vehicles is None:
                if "acceleration_non_compliance" in kwargs:
                    self.num_rl_vehicles = len(kwargs["acceleration_non_compliance"])
                else:
                    self.num_rl_vehicles = len(kwargs["acceleration_non_compliance"])

            if self.non_compliance_params["acceleration_type"] == "None":
                pass
            elif self.non_compliance_params["acceleration_type"] == "all_or_nothing":
                reward -= self.non_compliance_params["acceleration_gain"] * \
                          sum(np.array(kwargs["acceleration_non_compliance"]) != 0)
            elif self.non_compliance_params["acceleration_type"] == "linear":
                reward -= self.non_compliance_params["acceleration_gain"] * \
                          np.linalg.norm(kwargs["acceleration_non_compliance"], 1)
            elif self.non_compliance_params["acceleration_type"] == "quadratic":
                reward -= self.non_compliance_params["acceleration_gain"] * \
                          np.power(kwargs["acceleration_non_compliance"], 2)
            elif self.non_compliance_params["acceleration_type"] == "norm":
                reward -= self.non_compliance_params["acceleration_gain"] * \
                          np.linalg.norm(kwargs["acceleration_non_compliance"])
            elif self.non_compliance_params["acceleration_type"] == "function":
                reward -= self.non_compliance_params["acceleration_gain"] * \
                          self.non_compliance_params["acceleration_function"](kwargs["acceleration_non_compliance"])

            if self.non_compliance_params["lane_change_type"] == "None":
                pass
            elif self.non_compliance_params["lane_change_type"] == "all_or_nothing":
                reward -= self.non_compliance_params["lane_change_gain"] * \
                          sum(np.array(kwargs["lane_change_non_compliance"]) != 0)
            elif self.non_compliance_params["lane_change_type"] == "linear":
                reward -= self.non_compliance_params["lane_change_gain"] * \
                          np.linalg.norm(kwargs["lane_change_non_compliance"], 1)
            elif self.non_compliance_params["lane_change_type"] == "quadratic":
                reward -= self.non_compliance_params["lane_change_gain"] * \
                          np.power(kwargs["lane_change_non_compliance"], 2)
            elif self.non_compliance_params["lane_change_type"] == "norm":
                reward -= self.non_compliance_params["lane_change_gain"] * \
                          np.linalg.norm(kwargs["lane_change_non_compliance"])
            elif self.non_compliance_params["lane_change_type"] == "function":
                reward -= self.non_compliance_params["lane_change_gain"] * \
                          self.non_compliance_params["lane_change_function"](kwargs["lane_change_non_compliance"])
        else:
            kwargs["max_acceleration_penalty"] = 0
            kwargs["max_lane_change_penalty"] = 0

        if self.teach_non_collision:
            reward += self.make_reward_positive(kwargs["max_reward_penalty"], kwargs["max_lane_change_penalty"],
                                                kwargs["max_acceleration_penalty"])

        return reward

    def distance_from_target_velocity(self, state, rl_actions, **kwargs):
        """
        Reward function that determines the distance between the current velocity
        of vehicles in the fleet and the target velocity of the vehicles.

        :param state: numpy nd-array where the first state of states are the velocity
        of vehicles in the fleet
        :param rl_actions: array of actions performed by the rl vehicles
        :param kwargs: must contain "target_velocity"
        :return:
        """
        velocity = state[0]

        return - np.linalg.norm(velocity - kwargs["target_velocity"])