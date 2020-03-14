"""Contains a list of custom velocity controllers."""

from flow.controllers.base_controller import BaseController
import numpy as np


class FollowerStopper(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_des=15,
                 danger_edges=None):
        """Instantiate FollowerStopper."""
        BaseController.__init__(
            self, veh_id, car_following_params, delay=1.0,
            fail_safe='safe_velocity')

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5
        self.danger_edges = danger_edges if danger_edges else {}

    def find_intersection_dist(self, env):
        """Find distance to intersection.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.k.vehicle.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.network.edge_length(edge_id)
        relative_pos = env.k.vehicle.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)

        if self.v_des is None:
            return None

        if lead_id is None:
            v_cmd = self.v_des
        else:
            dx = env.k.vehicle.get_headway(self.veh_id)
            dv_minus = min(lead_vel - this_vel, 0)

            dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
            dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
            dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2
            v = min(max(lead_vel, 0), self.v_des)
            # compute the desired velocity
            if dx <= dx_1:
                v_cmd = 0
            elif dx <= dx_2:
                v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
            elif dx <= dx_3:
                v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                        / (dx_3 - dx_2)
            else:
                v_cmd = self.v_des

        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "":
            return None

        if self.find_intersection_dist(env) <= 10 and \
                env.k.vehicle.get_edge(self.veh_id) in self.danger_edges or \
                env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            return None
        else:
            # compute the acceleration from the desired velocity
            return (v_cmd - this_vel) / env.sim_step


class NonLocalFollowerStopper(FollowerStopper):
    """Follower stopper that uses the average system speed to compute its acceleration."""

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        self.v_des = np.mean(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

        if self.v_des is None:
            return None

        if lead_id is None:
            v_cmd = self.v_des
        else:
            dx = env.k.vehicle.get_headway(self.veh_id)
            dv_minus = min(lead_vel - this_vel, 0)

            dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus ** 2
            dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus ** 2
            dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus ** 2
            v = min(max(lead_vel, 0), self.v_des)
            # compute the desired velocity
            if dx <= dx_1:
                v_cmd = 0
            elif dx <= dx_2:
                v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
            elif dx <= dx_3:
                v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                        / (dx_3 - dx_2)
            else:
                v_cmd = self.v_des

        edge = env.k.vehicle.get_edge(self.veh_id)

        if edge == "":
            return None

        if self.find_intersection_dist(env) <= 10 and \
                env.k.vehicle.get_edge(self.veh_id) in self.danger_edges or \
                env.k.vehicle.get_edge(self.veh_id)[0] == ":":
            return None
        else:
            # compute the acceleration from the desired velocity
            return (v_cmd - this_vel) / env.sim_step


class PISaturation(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    car_following_params : flow.core.params.SumoCarFollowingParams
        object defining sumo-specific car-following parameters
    """

    def __init__(self, veh_id, car_following_params):
        """Instantiate PISaturation."""
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.alpha = 0
        self.beta = 1 - 0.5 * self.alpha
        self.U = 0
        self.v_target = 0
        self.v_cmd = 0

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        dx = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel
        dx_s = max(2 * dv, 4)

        # update the AV's velocity history
        self.v_history.append(this_vel)

        if len(self.v_history) == int(38 / env.sim_step):
            del self.v_history[0]

        # update desired velocity values
        v_des = np.mean(self.v_history)
        v_target = v_des + self.v_catch \
            * min(max((dx - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((dx - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)
