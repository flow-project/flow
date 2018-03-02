import numpy as np
import collections


class BaseController:
    def __init__(self, veh_id, controller_params):
        """Base class for flow-controlled acceleration behavior.

        Instantiates a controller and forces the user to pass a
        maximum acceleration to the controller. Provides the method
        safe_action to ensure that controls are never made that could
        cause the system to crash.

        Attributes
        ----------
        veh_id: string
            ID of the vehicle this controller is used for
        controller_params: dict
            Dictionary that optionally contains 'delay', the delay, and must
            contain 'max_deaccel', the maximum deceleration as well as all
            other parameters that dictate the driving behavior.
        """
        self.veh_id = veh_id
        self.controller_params = controller_params
        self.sumo_controller = False

        # magnitude of gaussian noise
        self.accel_noise = controller_params.get("noise", 0)

        # delay used by the safe_velocity failsafe
        self.delay = controller_params.get("delay", 0)

        # longitudinal failsafe used by the vehicle
        self.fail_safe = controller_params.get("fail_safe", None)

        # max deaccel should always be a positive
        self.max_deaccel = abs(controller_params.get('max_deaccel', 5))

        # acceleration queue, for time delayed actions
        self.accel_queue = collections.deque()

    def uses_sumo(self):
        return self.sumo_controller

    def reset_delay(self):
        self.accel_queue.clear()

    def get_accel(self, env):
        """Returns the acceleration of the controller"""
        raise NotImplementedError

    def get_action(self, env):
        """Converts the get_accel() acceleration into an action.

        If no acceleration is specified, the action returns a None as well,
        signifying that sumo should control the accelerations for the current
        time step.

        This method also augments the controller with the desired level of
        stochastic noise, and utlizes the "instantaneous" or "safe_velocity"
        failsafes if requested.

        Parameters
        ----------
        env: Env Type
            state of the environment at the current time step

        Returns
        -------
        action: float
            the modified form of the acceleration
        """
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
        """
        Instantaneously stops the car if there is a change of colliding into
        the leading vehicle in the next step

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step
        action: float
            requested acceleration action

        Returns
        -------
        safe_action: float
            the requested action if it does not lead to a crash; and a stopping
            action otherwise
        """
        # if there is only one vehicle in the network, all actions are safe
        if env.vehicles.num_vehicles == 1:
            return action

        lead_id = env.vehicles.get_leader(self.veh_id)

        # if there is no other vehicle in the lane, all actions are safe
        if lead_id is None:
            return action

        this_vel = env.vehicles.get_speed(self.veh_id)
        sim_step = env.sim_step
        next_vel = this_vel + action * sim_step
        h = env.vehicles.get_headway(self.veh_id)

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
        """Performs the "safe_velocity" failsafe action.

        Checks if the computed acceleration would put us above safe velocity.
        If it would, output the acceleration that would put at to safe
        velocity.

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step
        action: float
            requested acceleration action

        Returns
        -------
        safe_action: float
            the requested action clipped by the safe velocity
        """
        if env.vehicles.num_vehicles == 1:
            # if there is only one vehicle in the network, all actions are safe
            return action
        else:
            safe_velocity = self.safe_velocity(env)

            this_vel = env.vehicles.get_speed(self.veh_id)
            sim_step = env.sim_step

            if this_vel + action * sim_step > safe_velocity:
                return (safe_velocity - this_vel)/sim_step
            else:
                return action

    def safe_velocity(self, env):
        """Finds maximum velocity such that if the lead vehicle were to stop
        entirely, we can bring the following vehicle to rest at the point at
        which the headway is zero.

        Parameters
        ----------
        env: Environment type
            current environment, which contains information of the state of the
            network at the current time step

        Returns
        -------
        safe_velocity: float
            maximum safe velocity given a maximum deceleration and delay in
            performing the breaking action
        """
        lead_id = env.vehicles.get_leader(self.veh_id)
        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)

        h = env.vehicles.get_headway(self.veh_id)
        dv = lead_vel - this_vel

        v_safe = 2 * h / env.sim_step + dv - this_vel * (2 * self.delay)

        return v_safe
