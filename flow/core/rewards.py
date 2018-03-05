"""
This script contains of series of reward functions that can be used to train autonomous vehicles
"""

import numpy as np


def desired_velocity(env, fail=False):
    """
    A reward function used to encourage high system-level velocity.

    This function measures the deviation of a system of vehicles from a user-specified desired velocity,
    peaking when all vehicles in the ring are set to this desired velocity. Moreover, in order to ensure that
    the reward function naturally punishing the early termination of rollouts due to collisions or other failures,
    the function is formulated as a mapping $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}_{\geq 0}$.
    This is done by subtracting the deviation of the system from the desired velocity from the peak allowable
    deviation from the desired velocity. Additionally, since the velocity of vehicles are unbounded above, the
    reward is bounded below by zero, to ensure nonnegativity.

    :param env {SumoEnvironment type} - the environment variable, which contains information on the current
           state of the system.
    :param fail {bool} - specifies if any crash or other failure occurred in the system
    """
    vel = np.array(env.vehicles.get_speed(env.vehicles.get_ids()))
    num_vehicles = env.vehicles.num_vehicles

    if any(vel < -100) or fail:
        return 0.

    max_cost = np.array([env.env_params.additional_params["target_velocity"]] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - env.env_params.additional_params["target_velocity"]
    cost = np.linalg.norm(cost)

    return max(max_cost - cost, 0)


def rl_forward_progress(env, fail=False, gain = 0.1):
    """
        A reward function used to slightly rewards the RL vehicles travelling forward
        to help with sparse problems

        :param env {SumoEnvironment type} - the environment variable, which contains information on the current
               state of the system.
        :param fail {bool} - specifies if any crash or other failure occurred in the system
        :param gain {float} - specifies how much to reward the RL vehicles
        """
    rl_velocity = env.vehicles.get_speed(env.vehicles.get_rl_ids())
    rl_norm_vel = np.linalg.norm(rl_velocity, 1)
    return rl_norm_vel*gain


def boolean_action_penalty(discrete_actions, gain=1.0):
    """ Penalize boolean actions that indicate a switch"""
    return -gain*np.sum(discrete_actions)


def min_delay(state=None, actions=None, **kwargs):
    """
    A reward function used to encourage minimization of total delay in the 
    system. Distance travelled is used as a scaled value of delay. 
    
    This function measures the deviation of a system of vehicles
    from all the vehicles smoothly travelling at a fixed speed to their destinations.

    Note: state[0] MUST BE VELOCITY
    """

    vel = state[0]

    if any(vel < -100) or kwargs["fail"]:
        return 0.
    v_top = kwargs["target_velocity"]
    time_step = kwargs["time_step"]

    max_cost = time_step * sum(vel.shape)
    cost = time_step * sum((v_top - vel) / v_top)

    return max(max_cost - cost, 0)

def min_delay(env, fail=False):
    """
    A reward function used to encourage minimization of total delay in the
    system. Distance travelled is used as a scaled value of delay.

    This function measures the deviation of a system of vehicles
    from all the vehicles smoothly travelling at a fixed speed to their destinations.
    """

    vel = np.array(env.vehicles.get_speed())

    vel = vel[vel >= -1e-6]
    v_top = env.max_speed
    time_step = env.sim_step

    max_cost = time_step * sum(vel.shape)
    cost = time_step * sum((v_top - vel) / v_top)
    return max(max_cost - cost, 0)


def penalize_tl_changes(env, actions, gain=1, fail=False):
    delay = min_delay(env, fail)
    action_penalty = gain * np.sum(actions)
    return delay - action_penalty

def penalize_headway_variance(vehicles, vids, normalization=1, penalty_gain=1,
                              penalty_exponent=1):
    """
    A reward function used to train rl vehicles to encourage large
    headways among a pre-specified list of vehicles vids.

    :param vehicles {dict} - contains the state of all vehicles in the
    network (generally self.vehicles)
    :param vids: {list} - list of ids for vehicles
    :param normalization {float} - constant for scaling (down) the headways
    :param penalty_gain {float} - sets the penalty for each vehicle between
    0 and this value
    :param penalty_exponent {float} - used to allow exponential punishing of
    smaller headways
    :return: a (non-negative) penalty on vehicles whose headway is below the
    headway_threshold
    """
    headways = penalty_gain * np.power(np.array(
        [vehicles.get_headway(veh_id) / normalization for veh_id in vids]),
        penalty_exponent)
    return -np.var(headways)

def punish_small_rl_headways(vehicles, rl_ids, headway_threshold, penalty_gain=1, penalty_exponent=1):
    """
    A reward function used to train rl vehicles to avoid small headways.

    :param vehicles {dict} - contains the state of all vehicles in the network (generally self.vehicles)
    :param rl_ids: {list} - list of ids for rl vehicles in the network (generally self.rl_ids)
    :param headway_threshold {float} - the maximum headway allowed for rl vehicles before being penalized
    :param penalty_gain {float} - sets the penalty for each rl vehicle between 0 and this value
    :param penalty_exponent {float} - used to allow exponential punishing of smaller headways
    :return: a (non-negative) penalty on rl vehicles whose headway is below the headway_threshold
    """
    headway_penalty = 0
    for veh_id in rl_ids:
        if vehicles.get_headway(veh_id) < headway_threshold:
            headway_penalty += (((headway_threshold - vehicles.get_headway(veh_id)) / headway_threshold)
                                ** penalty_exponent) * penalty_gain

    # in order to keep headway penalty (and thus reward function) positive
    max_headway_penalty = len(rl_ids) * penalty_gain

    # return max_headway_penalty - headway_penalty
    return -np.abs(headway_penalty)


def punish_rl_lane_changes(vehicles, rl_ids, penalty=1):
    """
    A reward function that minimizes lane changes by producing a penalty every time an rl vehicle performs one.

    :param vehicles {dict} - contains the state of all vehicles in the network (generally self.vehicles)
    :param penalty {float} - penalty imposed on the reward function every time a
    :return:
    """
    total_lane_change_penalty = 0
    for veh_id in rl_ids:
        if vehicles[veh_id]["last_lc"] == self.timer:  # FIXME
            total_lane_change_penalty -= penalty

    return total_lane_change_penalty
