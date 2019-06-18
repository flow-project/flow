"""A series of reward functions."""

import numpy as np


def desired_velocity(env, fail=False, edge_list=None):
    r"""Encourage proximity to a desired velocity.

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

    Returns
    -------
    float
        reward value
    """
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return max(max_cost - cost, 0) / (max_cost + eps)


def average_velocity(env, fail=False):
    """Encourage proximity to an average velocity.

    This reward function returns the average velocity of all
    vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) == 0:
        return 0.

    return np.mean(vel)


def rl_forward_progress(env, gain=0.1):
    """Rewared function used to reward the RL vehicles for travelling forward.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        specifies how much to reward the RL vehicles

    Returns
    -------
    float
        reward value
    """
    rl_velocity = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
    rl_norm_vel = np.linalg.norm(rl_velocity, 1)
    return rl_norm_vel * gain


def boolean_action_penalty(discrete_actions, gain=1.0):
    """Penalize boolean actions that indicate a switch."""
    return gain * np.sum(discrete_actions)


def min_delay(env):
    """Reward function used to encourage minimization of total delay.

    This function measures the deviation of a system of vehicles from all the
    vehicles smoothly travelling at a fixed speed to their destinations.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.scenario.speed_limit(edge)
        for edge in env.k.scenario.get_edge_list())
    time_step = env.sim_step

    max_cost = time_step * sum(vel.shape)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return max((max_cost - cost) / (max_cost + eps), 0)


def min_delay_unscaled(env):
    """Return the average delay for all vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.scenario.speed_limit(edge)
        for edge in env.k.scenario.get_edge_list())
    time_step = env.sim_step

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return cost / (env.k.vehicle.num_vehicles + eps)


def penalize_standstill(env, gain=1):
    """Reward function that penalizes vehicle standstill.

    Is it better for this to be:
        a) penalize standstill in general?
        b) multiplicative based on time that vel=0?

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        multiplicative factor on the action penalty

    Returns
    -------
    float
        reward value
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_standstill = len(vel[vel == 0])
    penalty = gain * num_standstill
    return -penalty


def penalize_near_standstill(env, thresh=0.3, gain=1):
    """Reward function which penalizes vehicles at a low velocity.

    This reward function is used to penalize vehicles below a
    specified threshold. This assists with discouraging RL from
    gamifying a scenario, which can result in standstill behavior
    or similarly bad, near-zero velocities.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
    thresh : float
        the velocity threshold below which penalties are applied
    gain : float
        multiplicative factor on the action penalty
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    penalize = len(vel[vel < thresh])
    penalty = gain * penalize
    return -penalty


def penalize_headway_variance(vehicles,
                              vids,
                              normalization=1,
                              penalty_gain=1,
                              penalty_exponent=1):
    """Reward function used to train rl vehicles to encourage large headways.

    Parameters
    ----------
    vehicles : flow.core.kernel.vehicle.KernelVehicle
        contains the state of all vehicles in the network (generally
        self.vehicles)
    vids : list of str
        list of ids for vehicles
    normalization : float, optional
        constant for scaling (down) the headways
    penalty_gain : float, optional
        sets the penalty for each vehicle between 0 and this value
    penalty_exponent : float, optional
        used to allow exponential punishing of smaller headways
    """
    headways = penalty_gain * np.power(
        np.array(
            [vehicles.get_headway(veh_id) / normalization
             for veh_id in vids]), penalty_exponent)
    return -np.var(headways)


def punish_rl_lane_changes(env, penalty=1):
    """Penalize an RL vehicle performing lane changes.

    This reward function is meant to minimize the number of lane changes and RL
    vehicle performs.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    penalty : float, optional
        penalty imposed on the reward function for any rl lane change action
    """
    total_lane_change_penalty = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.timer:
            total_lane_change_penalty -= penalty

    return total_lane_change_penalty
