"""
This script contains of series of reward functions that can be used to train autonomous vehicles
"""

import numpy as np
import pdb


def desired_velocity(state=None, actions=None, **kwargs):
    """
    A reward function used to encourage high system-level velocity.

    This function measures the deviation of a system of vehicles from a user-specified desired velocity,
    peaking when all vehicles in the ring are set to this desired velocity. Moreover, in order to ensure that
    the reward function naturally punishing the early termination of rollouts due to collisions or other failures,
    the function is formulated as a mapping $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}_{\geq 0}$.
    This is done by subtracting the deviation of the system from the desired velocity from the peak allowable
    deviation from the desired velocity. Additionally, since the velocity of vehicles are unbounded above, the
    reward is bounded below by zero, to ensure nonnegativity.

    Note: state[0] MUST BE VELOCITY
    """
    num_vehicles = len(state[0])
    vel = state[0]

    if any(vel < -100) or kwargs["fail"]:
        return 0.

    max_cost = np.array([kwargs["target_velocity"]] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - kwargs["target_velocity"]
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


def distance_traveled(state=None, actions=None, **kwargs):
    # TODO
    pass


def emission(state=None, actions=None, **kwargs):
    # TODO
    pass

