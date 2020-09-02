"""Contains the base acceleration controller class."""

from abc import ABCMeta, abstractmethod
import numpy as np
import math


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
        self.max_speed = car_following_params.controller_params['maxSpeed']

        self.car_following_params = car_following_params

        self._is_highway_i210 = None

    @abstractmethod
    def get_accel(self, env):
        """Return the acceleration of the controller."""
        pass

    @abstractmethod
    def get_custom_accel(self, this_vel, lead_vel, h):
        """Return the custom computed acceleration of the controller.

        This method computes acceleration based on custom state information,
        while get_accel() method compute acceleration based on the current state
        information that are obtained from the environment.

        Parameters
        ----------
        this_vel : float
            this vehicle's velocity
        lead_vel : float
            leading vehicle's velocity
        h : float
            headway to leading vehicle

        Returns
        -------
        float
            the custom acceleration of the controller
        """
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
        if self._is_highway_i210 is None:
            from flow.networks import I210SubNetwork
            from flow.networks import HighwayNetwork
            self._is_highway_i210 = \
                isinstance(env.k.network.network, I210SubNetwork) or \
                isinstance(env.k.network.network, HighwayNetwork)

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
        if env.k.vehicle.get_edge(self.veh_id)[0] == ":" \
                and not self._is_highway_i210:
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
        accel = self.compute_failsafe(accel, env)

        env.k.vehicle.update_accel(self.veh_id, accel, noise=True, failsafe=True)
        return accel

    def compute_failsafe(self, accel, env):
        """Take in an acceleration and compute the resultant safe acceleration."""
        for failsafe in self.failsafes:
            accel = failsafe(env, accel)
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
        instantaneously, we can bring the following vehicle to rest at the point at
        which the headway is zero.

        WARNINGS:
        1. We assume the lead vehicle has the same deceleration capabilities as our vehicles
        2. We solve for this value using the discrete time approximation to the dynamics. We assume that the
           integration scheme induces positive error in the position, which leads to a slightly more conservative
           driving behavior than the continuous time approximation would induce. However, the continuous time
           safety rule would not be strictly safe.

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
        if not lead_id:
            return 1000.0
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        max_decel = self.max_deaccel
        lead_control = env.k.vehicle.get_acc_controller(lead_id)
        lead_max_deaccel = lead_control.max_deaccel

        h = env.k.vehicle.get_headway(self.veh_id)
        assert (h > 0), print('the headway is less than zero! Seems wrong.')
        min_gap = self.car_following_params.controller_params['minGap']

        is_ballistic = env.sim_params.use_ballistic
        just_inserted = self.veh_id in env.k.vehicle.get_departed_ids()
        brake_distance = self.brake_distance(lead_vel, max(max_decel, lead_max_deaccel),
                                             self.delay, is_ballistic, env.sim_step)
        v_safe = self.maximum_safe_stop_speed(h + brake_distance - min_gap, this_vel, just_inserted,
                                              is_ballistic, env.sim_step)

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
        this_lane = env.k.vehicle.get_lane(self.veh_id)
        edge_speed_limit = env.k.network.get_max_speed(this_edge, this_lane)
        veh_speed_limit = self.max_speed
        edge_speed_limit = min(veh_speed_limit, edge_speed_limit)

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

    def brake_distance(self, speed, max_deaccel, delay, is_ballistic, sim_step):
        """Return the distance needed to come to a full stop if braking as hard as possible.

        Parameters
        ----------
        speed : float
            ego speed
        max_deaccel : float
            maximum deaccel of the vehicle
        delay : float
            the delay before an action is executed
        is_ballistic : bool
            whether the integration stop is ballistic
        sim_step : float
            size of simulation step

        Returns
        -------
        float
            the distance required to stop
        """
        if is_ballistic:
            if speed <= 0:
                return 0.0
            else:
                return speed * (delay + 0.5 * speed / max_deaccel)
        else:
            # how much we can reduce the speed in each timestep
            speedReduction = max_deaccel * sim_step
            # how many steps to get the speed to zero
            steps_to_zero = int(speed / speedReduction)
            return sim_step * (steps_to_zero * speed - speedReduction * steps_to_zero * (steps_to_zero + 1) / 2) + \
                speed * delay

    def maximum_safe_stop_speed(self, brake_distance, speed, is_inserted, is_ballistic, sim_step):
        """Compute the maximum speed that you can travel at and guarantee no collision.

        Parameters
        ----------
        brake_distance : float
            total distance the vehicle has before it must be at a full stop
        speed : float
            current vehicle speed
        is_inserted : bool
            whether the vehicle has just entered the network
        is_ballistic : bool
            whether the integrator is ballistic
        sim_step : float
            simulation step size in seconds

        Returns
        -------
        v_safe : float
            maximum speed that can be travelled at without crashing
        """
        if is_ballistic:
            v_safe = self.maximum_safe_stop_speed_ballistic(brake_distance, speed, is_inserted,
                                                            sim_step)
        else:
            v_safe = self.maximum_safe_stop_speed_euler(brake_distance, sim_step)
        return v_safe

    def maximum_safe_stop_speed_euler(self, brake_distance, sim_step):
        """Compute the maximum speed that you can travel at and guarantee no collision for euler integration.

        Parameters
        ----------
        brake_distance : float
            total distance the vehicle has before it must be at a full stop
        sim_step : float
            simulation step size in seconds

        Returns
        -------
        v_safe : float
            maximum speed that can be travelled at without crashing
        """
        if brake_distance <= 0:
            return 0.0

        speed_reduction = self.max_deaccel * sim_step

        s = sim_step
        t = self.delay

        # h = the distance that would be covered if it were possible to stop
        # exactly after gap and decelerate with max_deaccel every simulation step
        # h = 0.5 * n * (n-1) * b * s + n * b * t (solve for n)
        # n = ((1.0/2.0) - ((t + (pow(((s*s) + (4.0*((s*((2.0*h/b) - t)) + (t*t)))), (1.0/2.0))*sign/2.0))/s))
        sqrt_quantity = math.sqrt(
            ((s * s) + (4.0 * ((s * (2.0 * brake_distance / speed_reduction - t)) + (t * t))))) * -0.5
        n = math.floor(.5 - ((t + sqrt_quantity) / s))
        h = 0.5 * n * (n - 1) * speed_reduction * s + n * speed_reduction * t
        assert(h <= brake_distance + 1e-6)
        # compute the additional speed that must be used during deceleration to fix
        # the discrepancy between g and h
        r = (brake_distance - h) / (n * s + t)
        x = n * speed_reduction + r
        assert(x >= 0)
        return x

    def maximum_safe_stop_speed_ballistic(self, brake_distance, speed, is_inserted, sim_step):
        """Compute the maximum speed that you can travel at and guarantee no collision for ballistic integration.

        Parameters
        ----------
        brake_distance : float
            total distance the vehicle has before it must be at a full stop
        speed : float
            current vehicle speed
        is_inserted : bool
            whether the vehicle has just entered the network
        sim_step : float
            simulation step size in seconds

        Returns
        -------
        v_safe : float
            maximum speed that can be travelled at without crashing
        """
        # decrease gap slightly (to avoid passing end of lane by values of magnitude ~1e-12,
        # when exact stop is required)
        new_brake_gap = max(0., brake_distance - 1e-6)

        # (Leo) Note that in contrast to the Euler update, for the ballistic update
        # the distance covered in the coming step depends on the current velocity, in general.
        # one exception is the situation when the vehicle is just being inserted.
        # In that case, it will not cover any distance until the next timestep by convention.

        # We treat the latter case first:
        if is_inserted:
            # The distance covered with constant insertion speed v0 until time tau is given as
            # G1 = tau*v0
            # The distance covered between time tau and the stopping moment at time tau+v0/b is
            # G2 = v0^2/(2b),
            # where b is an assumed constant deceleration (= myDecel)
            # We solve g = G1 + G2 for v0:
            btau = self.max_deaccel * self.delay
            v0 = -btau + np.sqrt(btau * btau + 2 * self.max_deaccel * new_brake_gap)
            return v0

        # In the usual case during the driving task, the vehicle goes by
        # a current speed v0=v, and we seek to determine a safe acceleration a (possibly <0)
        # such that starting to break after accelerating with a for the time tau=self.delay
        # still allows us to stop in time.

        if self.delay == 0:
            tau = sim_step
        else:
            tau = self.delay
        v0 = max(0., speed)
        # We first consider the case that a stop has to take place within time tau
        if v0 * tau >= 2 * new_brake_gap:
            if new_brake_gap == 0:
                if v0 > 0.:
                    # indicate to brake as hard as possible
                    return -self.max_deaccel * sim_step
                else:
                    # stay stopped
                    return 0
            # In general we solve g = v0^2/(-2a), where the the rhs is the distance
            # covered until stop when breaking with a<0
            a = -v0 * v0 / (2 * new_brake_gap)
            return v0 + a * sim_step

        # The last case corresponds to a situation, where the vehicle may go with a positive
        # speed v1 = v0 + tau*a after time tau.
        # The distance covered until time tau is given as
        # G1 = tau*(v0+v1)/2
        # The distance covered between time tau and the stopping moment at time tau+v1/b is
        # G2 = v1^2/(2b),
        # where b is an assumed constant deceleration (= myDecel)
        # We solve g = G1 + G2 for v1>0:
        # <=> 0 = v1^2 + b*tau*v1 + b*tau*v0 - 2bg
        #  => v1 = -b*tau/2 + sqrt( (b*tau)^2/4 + b(2g - tau*v0) )

        btau2 = self.max_deaccel * tau / 2
        v1 = -btau2 + np.sqrt(btau2 * btau2 + self.max_deaccel * (2 * new_brake_gap - tau * v0))
        a = (v1 - v0) / tau
        return v0 + a * sim_step
