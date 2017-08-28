"""
This script contains of series of reward functions that can be used to train autonomous vehicles
"""

from cistar_dev.controllers.rlcontroller import RLController

import numpy as np
import pdb


# TODO: create local version (for moving bottleneck, ...)
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
    vel = np.array([env.vehicles[veh_id]["speed"] for veh_id in env.vehicles.keys()])
    num_vehicles = len(vel)

    if any(vel < -100) or fail:
        return 0.

    max_cost = np.array([env.env_params["target_velocity"]] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - env.env_params["target_velocity"]
    cost = np.linalg.norm(cost)

    return max(max_cost - cost, 0)


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

    max_cost = time_step*sum(vel.shape)
    cost = time_step*sum((v_top - vel)/v_top)

    return max(max_cost - cost, 0)


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
        if vehicles[veh_id]["headway"] < headway_threshold:
            headway_penalty += (((headway_threshold - vehicles[veh_id]["headway"]) / headway_threshold)
                                ** penalty_exponent) * penalty_gain

    # in order to keep headway penalty (and thus reward function) positive
    max_headway_penalty = len(rl_ids) * penalty_gain

    return max_headway_penalty - headway_penalty


def minimize_rl_lane_changes(vehicles, rl_ids, penalty=1):
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


def distance_traveled(state=None, actions=None, **kwargs):
    # TODO
    pass


def emission(state=None, actions=None, **kwargs):
    # TODO
    pass

