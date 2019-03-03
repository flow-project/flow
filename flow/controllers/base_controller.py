"""Contains the base acceleration controller class."""

import numpy as np


class BaseController:
    """Base class for flow-controlled acceleration behavior.

    Instantiates a controller and forces the user to pass a
    maximum acceleration to the controller. Provides the method
    safe_action to ensure that controls are never made that could
    cause the system to crash.

    Parameters
    ----------
    veh_id : str
        ID of the vehicle this controller is used for
    car_following_params : flow.core.params.SumoCarFollowingParams
        The underlying sumo model for car that will be overwritten. A Flow
        controller will override the behavior this sumo car following
        model; however, if control is ceded back to sumo, the vehicle will
        use these params. Ensure that accel / decel parameters that are
        specified to in this model are as desired.
    delay : int
        delay in applying the action (time)
    fail_safe : str
        Should be either "instantaneous" or "safe_velocity"
    noise : double
        variance of the gaussian from which to sample a noisy acceleration
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 delay=0,
                 fail_safe=None,
                 noise=0):
        """Instantiate the base class for acceleration behavior."""
        self.veh_id = veh_id

        # magnitude of gaussian noise
        self.accel_noise = noise

        # delay used by the safe_velocity failsafe
        self.delay = delay

        # longitudinal failsafe used by the vehicle
        self.fail_safe = fail_safe

        self.max_accel = car_following_params.controller_params['accel']
        # max deaccel should always be a positive
        self.max_deaccel = abs(car_following_params.controller_params['decel'])

        self.car_following_params = car_following_params

    def get_accel(self, env):
        """Return the acceleration of the controller."""
        raise NotImplementedError

    def get_action(self, env):
        """Convert the get_accel() acceleration into an action.

        If no acceleration is specified, the action returns a None as well,
        signifying that sumo should control the accelerations for the current
        time step.

        This method also augments the controller with the desired level of
        stochastic noise, and utlizes the "instantaneous" or "safe_velocity"
        failsafes if requested.

        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step

        Returns
        -------
        float
            the modified form of the acceleration
        """
        # this is to avoid abrupt decelerations when a vehicle has just entered
        # a network and it's data is still not subscribed
        if len(env.k.vehicle.get_edge(self.veh_id)) == 0:
            return None

        # this allows the acceleration behavior of vehicles in a junction be
        # described by sumo instead of an explicit model
        if env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            return None

        accel = self.get_accel(env)

        # if no acceleration is specified, let sumo take over for the current
        # time step
        if accel is None:
            return None

        # add noise to the accelerations, if requested
        if self.accel_noise > 0:
            accel += np.random.normal(0, self.accel_noise)

        # run the failsafes, if requested
        if self.fail_safe == 'instantaneous':
            accel = self.get_safe_action_instantaneous(env, accel)
        elif self.fail_safe == 'safe_velocity':
            accel = self.get_safe_velocity_action(env, accel)

        return accel

    def get_safe_action_instantaneous(self, env, action):
        """Perform the "instantaneous" failsafe action.

        Instantaneously stops the car if there is a change of colliding into
        the leading vehicle in the next step

        Parameters
        ----------
        env : flow.envs.Env
            current environment, which contains information of the state of the
            network at the current time step
        action : float
            requested acceleration action

        Returns
        -------
        float
            the requested action if it does not lead to a crash; and a stopping
            action otherwise
        """
        # if there is only one vehicle in the network, all actions are safe
        if env.k.vehicle.num_vehicles == 1:
            return action

        lead_id = env.k.vehicle.get_leader(self.veh_id)

        # if there is no other vehicle in the lane, all actions are safe
        if lead_id is None:
            return action

        this_vel = env.k.vehicle.get_speed(self.veh_id)
        sim_step = env.sim_step
        next_vel = this_vel + action * sim_step
        h = env.k.vehicle.get_headway(self.veh_id)

        if next_vel > 0:
            # the second and third terms cover (conservatively) the extra
            # distance the vehicle will cover before it fully decelerates
            if h < sim_step * next_vel + this_vel * 1e-3 + \
                    0.5 * this_vel * sim_step:
                # if the vehicle will crash into the vehicle ahead of it in the
                # next time step (assuming the vehicle ahead of it is not
                # moving), then stop immediately
                return -this_vel / sim_step
            else:
                # if the vehicle is not in danger of crashing, continue with
                # the requested action
                return action
        else:
            return action

    def get_safe_velocity_action(self, env, action):
        """Perform the "safe_velocity" failsafe action.

        Checks if the computed acceleration would put us above safe velocity.
        If it would, output the acceleration that would put at to safe
        velocity.

        Parameters
        ----------
        env : flow.envs.Env
            current environment, which contains information of the state of the
            network at the current time step
        action : float
            requested acceleration action

        Returns
        -------
        float
            the requested action clipped by the safe velocity
        """
        if env.k.vehicle.num_vehicles == 1:
            # if there is only one vehicle in the network, all actions are safe
            return action
        else:
            safe_velocity = self.safe_velocity(env)

            this_vel = env.k.vehicle.get_speed(self.veh_id)
            sim_step = env.sim_step

            if this_vel + action * sim_step > safe_velocity:
                if safe_velocity > 0:
                    return (safe_velocity - this_vel) / sim_step
                else:
                    return -this_vel / sim_step
            else:
                return action

    def safe_velocity(self, env):
        """Compute a safe velocity for the vehicles.

        Finds maximum velocity such that if the lead vehicle were to stop
        entirely, we can bring the following vehicle to rest at the point at
        which the headway is zero.

        Parameters
        ----------
        env : flow.envs.Env
            current environment, which contains information of the state of the
            network at the current time step

        Returns
        -------
        float
            maximum safe velocity given a maximum deceleration and delay in
            performing the breaking action
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        h = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel

        v_safe = 2 * h / env.sim_step + dv - this_vel * (2 * self.delay)

        return v_safe
