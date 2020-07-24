"""Contains the base acceleration controller class."""

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseController(metaclass=ABCMeta):
    """Base class for flow-controlled acceleration behavior.

    Instantiates a controller and forces the user to pass a
    maximum acceleration to the controller. Provides the method
    safe_action to ensure that controls are never made that could
    cause the system to crash.

    Usage
    -----
    >>> from flow.core.params import VehicleParams
    >>> from flow.controllers import IDMController
    >>> vehicles = VehicleParams()
    >>> vehicles.add("human", acceleration_controller=(IDMController, {}))

    Note: You can replace "IDMController" with any subclass controller of your
    choice.

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
    fail_safe : list of str or str
        List of failsafes which can be "instantaneous", "safe_velocity",
        "feasible_accel", or "obey_speed_limit". The order of applying the
        falsafes will be based on the order in the list.
    display_warnings : bool
        Flag for toggling on/off printing failsafe warnings to screen.
    noise : double
        variance of the gaussian from which to sample a noisy acceleration
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 delay=0,
                 fail_safe=None,
                 display_warnings=True,
                 noise=0):
        """Instantiate the base class for acceleration behavior."""
        self.veh_id = veh_id

        # magnitude of gaussian noise
        self.accel_noise = noise

        # delay used by the safe_velocity failsafe
        self.delay = delay

        # longitudinal failsafe used by the vehicle
        if isinstance(fail_safe, str):
            failsafe_list = [fail_safe]
        elif isinstance(fail_safe, list) or fail_safe is None:
            failsafe_list = fail_safe
        else:
            failsafe_list = None
            raise ValueError("fail_safe should be string or list of strings. Setting fail_safe to None\n")

        failsafe_map = {
            'instantaneous': self.get_safe_action_instantaneous,
            'safe_velocity': self.get_safe_velocity_action,
            'feasible_accel': lambda _, accel: self.get_feasible_action(accel),
            'obey_speed_limit': self.get_obey_speed_limit_action
        }
        self.failsafes = []
        if failsafe_list:
            for check in failsafe_list:
                if check in failsafe_map:
                    self.failsafes.append(failsafe_map.get(check))
                else:
                    raise ValueError('Skipping {}, as it is not a valid failsafe.'.format(check))

        self.display_warnings = display_warnings

        self.max_accel = car_following_params.controller_params['accel']
        # max deaccel should always be a positive
        self.max_deaccel = abs(car_following_params.controller_params['decel'])

        self.car_following_params = car_following_params

    @abstractmethod
    def get_accel(self, env):
        """Return the acceleration of the controller."""
        pass

    def get_action(self, env):
        """Convert the get_accel() acceleration into an action.

        If no acceleration is specified, the action returns a None as well,
        signifying that sumo should control the accelerations for the current
        time step.

        This method also augments the controller with the desired level of
        stochastic noise, and utlizes the "instantaneous", "safe_velocity",
        "feasible_accel", and/or "obey_speed_limit" failsafes if requested.

        Parameters
        ----------
        env : flow.envs.Env
            state of the environment at the current time step

        Returns
        -------
        float
            the modified form of the acceleration
        """
        # clear the current stored accels of this vehicle to None
        env.k.vehicle.update_accel(self.veh_id, None, noise=False, failsafe=False)
        env.k.vehicle.update_accel(self.veh_id, None, noise=False, failsafe=True)
        env.k.vehicle.update_accel(self.veh_id, None, noise=True, failsafe=False)
        env.k.vehicle.update_accel(self.veh_id, None, noise=True, failsafe=True)

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

        # store the acceleration without noise to each vehicle
        # run fail safe if requested
        env.k.vehicle.update_accel(self.veh_id, accel, noise=False, failsafe=False)
        accel_no_noise_with_failsafe = accel

        for failsafe in self.failsafes:
            accel_no_noise_with_failsafe = failsafe(env, accel_no_noise_with_failsafe)

        env.k.vehicle.update_accel(self.veh_id, accel_no_noise_with_failsafe, noise=False, failsafe=True)

        # add noise to the accelerations, if requested
        if self.accel_noise > 0:
            accel += np.sqrt(env.sim_step) * np.random.normal(0, self.accel_noise)
        env.k.vehicle.update_accel(self.veh_id, accel, noise=True, failsafe=False)

        # run the fail-safes, if requested
        for failsafe in self.failsafes:
            accel = failsafe(env, accel)

        env.k.vehicle.update_accel(self.veh_id, accel, noise=True, failsafe=True)
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
                if self.display_warnings:
                    print(
                        "=====================================\n"
                        "Vehicle {} is about to crash. Instantaneous acceleration "
                        "clipping applied.\n"
                        "=====================================".format(self.veh_id))

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
            maximum safe velocity given a maximum deceleration, delay in
            performing the breaking action, and speed limit
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        h = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel

        v_safe = 2 * h / env.sim_step + dv - this_vel * (2 * self.delay)

        # check for speed limit  FIXME: this is not called
        # this_edge = env.k.vehicle.get_edge(self.veh_id)
        # edge_speed_limit = env.k.network.speed_limit(this_edge)

        if this_vel > v_safe:
            if self.display_warnings:
                print(
                    "=====================================\n"
                    "Speed of vehicle {} is greater than safe speed. Safe velocity "
                    "clipping applied.\n"
                    "=====================================".format(self.veh_id))

        return v_safe

    def get_obey_speed_limit_action(self, env, action):
        """Perform the "obey_speed_limit" failsafe action.

        Checks if the computed acceleration would put us above edge speed limit.
        If it would, output the acceleration that would put at the speed limit
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
            the requested action clipped by the speed limit
        """
        # check for speed limit
        this_edge = env.k.vehicle.get_edge(self.veh_id)
        edge_speed_limit = env.k.network.speed_limit(this_edge)

        this_vel = env.k.vehicle.get_speed(self.veh_id)
        sim_step = env.sim_step

        if this_vel + action * sim_step > edge_speed_limit:
            if edge_speed_limit > 0:
                if self.display_warnings:
                    print(
                        "=====================================\n"
                        "Speed of vehicle {} is greater than speed limit. Obey "
                        "speed limit clipping applied.\n"
                        "=====================================".format(self.veh_id))
                return (edge_speed_limit - this_vel) / sim_step
            else:
                return -this_vel / sim_step
        else:
            return action

    def get_feasible_action(self, action):
        """Perform the "feasible_accel" failsafe action.

        Checks if the computed acceleration would put us above maximum
        acceleration or deceleration. If it would, output the acceleration
        equal to maximum acceleration or deceleration.

        Parameters
        ----------
        action : float
            requested acceleration action

        Returns
        -------
        float
            the requested action clipped by the feasible acceleration or
            deceleration.
        """
        if action > self.max_accel:
            action = self.max_accel

            if self.display_warnings:
                print(
                    "=====================================\n"
                    "Acceleration of vehicle {} is greater than the max "
                    "acceleration. Feasible acceleration clipping applied.\n"
                    "=====================================".format(self.veh_id))

        if action < -self.max_deaccel:
            action = -self.max_deaccel

            if self.display_warnings:
                print(
                    "=====================================\n"
                    "Deceleration of vehicle {} is greater than the max "
                    "deceleration. Feasible acceleration clipping applied.\n"
                    "=====================================".format(self.veh_id))

        return action
