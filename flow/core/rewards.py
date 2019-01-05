"""This script contains of series of reward functions."""

import numpy as np


def desired_velocity(env, fail=False, edge_list=None):
    """Encourage proximity to a desired velocity.

    This function measures the deviation of a system of vehicles from a
    user-specified desired velocity peaking when all vehicles in the ring
    are set to this desired velocity. Moreover, in order to ensure that the
    reward function naturally punishing the early termination of rollouts due
    to collisions or other failures, the function is formulated as a mapping
    :math:`r: \\mathcal{S} \\times \\mathcal{A}
    \\rightarrow \\mathbb{R}_{\\geq 0}`.
    This is done by subtracting the deviation of the system from the
    desired velocity from the peak allowable deviation from the desired
    velocity. Additionally, since the velocity of vehicles are
    unbounded above, the reward is bounded below by zero,
    to ensure nonnegativity.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system
    edge_list : list  of str, optional
        list of edges the reward is computed over. If no edge_list is defined,
        the reward is computed over all edges
    """
    if edge_list is None:
        veh_ids = env.vehicles.get_ids()
    else:
        veh_ids = env.vehicles.get_ids_by_edge(edge_list)

    vel = np.array(env.vehicles.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    return max(max_cost - cost, 0) / max_cost


def average_velocity(env, fail=False):
    vel = np.array(env.vehicles.get_speed(env.vehicles.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) == 0:
        return 0.

    return np.mean(vel)


def total_velocity(env, fail=False):
    vel = np.array(env.vehicles.get_speed(env.vehicles.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) != 0:
        return sum(vel)


def reward_density(env):
    return env.vehicles.get_num_arrived() / env.sim_step


def rl_forward_progress(env, gain=0.1):
    """A reward function used to reward the RL vehicles travelling forward.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain: float
        specifies how much to reward the RL vehicles
    """
    rl_velocity = env.vehicles.get_speed(env.vehicles.get_rl_ids())
    rl_norm_vel = np.linalg.norm(rl_velocity, 1)
    return rl_norm_vel * gain


def boolean_action_penalty(discrete_actions, gain=1.0):
    """Penalize boolean actions that indicate a switch"""
    return gain * np.sum(discrete_actions)


def min_delay(env):
    """A reward function used to encourage minimization of total delay.

    This function measures the deviation of a system of vehicles from all the
    vehicles smoothly travelling at a fixed speed to their destinations.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    """

    vel = np.array(env.vehicles.get_speed(env.vehicles.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.scenario.speed_limit(edge)
        for edge in env.k.scenario.get_edge_list())
    time_step = env.sim_step

    max_cost = time_step * sum(vel.shape)
    try:
        cost = time_step * sum((v_top - vel) / v_top)
        return max((max_cost - cost) / max_cost, 0)
    except ZeroDivisionError:
        return 0


def min_delay_unscaled(env):
    """The average delay for all vehicles in the system

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    """

    vel = np.array(env.vehicles.get_speed(env.vehicles.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.scenario.speed_limit(edge)
        for edge in env.k.scenario.get_edge_list())
    time_step = env.sim_step

    cost = time_step * sum((v_top - vel) / v_top)
    return cost / len(env.vehicles.get_ids())


def penalize_standstill(env, gain=1):
    """A reward function that penalizes vehicle standstill

    Is it better for this to be:
        a) penalize standstill in general?
        b) multiplicative based on time that vel=0?

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        multiplicative factor on the action penalty
    """
    veh_ids = env.vehicles.get_ids()
    vel = np.array(env.vehicles.get_speed(veh_ids))
    num_standstill = len(vel[vel == 0])
    penalty = gain * num_standstill
    return -penalty


def penalize_near_standstill(env, thresh=0.3, gain=1):
    veh_ids = env.vehicles.get_ids()
    vel = np.array(env.vehicles.get_speed(veh_ids))
    penalize = len(vel[vel < thresh])
    penalty = gain * penalize
    return -penalty


def penalize_headway_variance(vehicles,
                              vids,
                              normalization=1,
                              penalty_gain=1,
                              penalty_exponent=1):
    """A reward function used to train rl vehicles to encourage large headways

    Parameters
    ----------
    vehicles: flow.core.vehicles.Vehicles
        contains the state of all vehicles in the network (generally
        self.vehicles)
    vids: list of str
        list of ids for vehicles
    normalization: float, optional
        constant for scaling (down) the headways
    penalty_gain: float, optional
        sets the penalty for each vehicle between 0 and this value
    penalty_exponent: float, optional
        used to allow exponential punishing of smaller headways
    """
    headways = penalty_gain * np.power(
        np.array(
            [vehicles.get_headway(veh_id) / normalization
             for veh_id in vids]), penalty_exponent)
    return -np.var(headways)


def punish_small_rl_headways(env,
                             headway_threshold=5,
                             penalty_gain=1,
                             penalty_exponent=1):
    """A reward function used to train rl vehicles to avoid small headways.

    A penalty is issued whenever rl vehicles are below a pre-defined desired
    headway.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    headway_threshold: float
        the maximum headway allowed for rl vehicles before being penalized
    penalty_gain: float, optional
        sets the penalty for each rl vehicle between 0 and this value
    penalty_exponent: float, optional
        used to allow exponential punishing of smaller headways
    """
    headway_penalty = 0
    for veh_id in env.vehicles.get_rl_ids():
        if env.vehicles.get_headway(veh_id) < headway_threshold:
            headway_penalty += \
                (((headway_threshold - env.vehicles.get_headway(veh_id)) /
                  headway_threshold) ** penalty_exponent) * penalty_gain

    return -np.abs(headway_penalty)


def punish_rl_lane_changes(env, penalty=1):
    """Penalize an RL vehicle performing lane changes.

    This reward function is meant to minimize the number of lane changes and RL
    vehicle performs.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    penalty : float, optional
        penalty imposed on the reward function for any rl lane change action
    """
    total_lane_change_penalty = 0
    for veh_id in env.vehicles.get_rl_ids():
        if env.vehicles.get_state(veh_id, 'last_lc') == env.timer:
            total_lane_change_penalty -= penalty

    return total_lane_change_penalty


def punish_queues_in_lane(env, edge, lane, penalty_gain=1, penalty_exponent=1):
    """Punish queues in certain lanes of edge '3'.

    TODO: specify what scenario this is used by

    Parameters
    ----------
    env : flow.envs.Env
        contains the state of the environment at a time-step
    edge: str
        The edge on which to penalize queues
    lane : int
        The lane in which to penalize queues
    penalty_gain : int, optional
        Multiplier on number of cars in the lane
    penalty_exponent : int, optional
        Exponent on number of cars in the lane

    Returns
    -------
    float
        total reward (in this case a negative cost) corresponding to the queues
        in the lane in question
    """
    # IDs of all vehicles in passed-in lane
    lane_ids = [
        veh_id for veh_id in env.vehicles.get_ids_by_edge(edge)
        if env.vehicles.get_lane(veh_id) == lane
    ]

    return -1 * (len(lane_ids) ** penalty_exponent) * penalty_gain


def reward_rl_opening_headways(env, reward_gain=0.1, reward_exponent=1):
    """Reward RL vehicles opening large headways.

    Parameters
    ----------
    env : flow.envs.Env
        contains the state of the environment at a time-step
    reward_gain : int, optional
        Multiplicative gain on reward
    reward_exponent : int, optional
        Exponent gain on reward

    Returns
    -------
    float
        Reward value
    """
    total_reward = 0
    for rl_id in env.vehicles.get_rl_ids():
        follower_id = env.vehicles.get_follower(rl_id)
        if not follower_id:
            continue
        follower_headway = env.vehicles.get_headway(follower_id)
        if follower_headway < 0:
            continue
        total_reward += follower_headway ** reward_exponent

    return total_reward * reward_gain
