from flow.controllers.base_controller import BaseController
import numpy as np


class FollowerStopper(BaseController):

    def __init__(self, veh_id, v_des=15, max_accel=1.0):
        """Inspired by Dan Work's... work:

		Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments
		https://arxiv.org/abs/1705.01693

        Parameters
        ----------
        veh_id: str
            unique vehicle identifier
        v_des: float, optional
            desired speed of the vehicles (m/s)
        max_accel: float, optional
            maximum achievable acceleration by the vehicle (m/s^2)
        """
        controller_params = {"delay": 1.0, "max_deaccel": 1.5,
                             "noise": 0, "fail_safe": "safe_velocity"}
        BaseController.__init__(self, veh_id, controller_params)

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = max_accel

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5

    def get_accel(self, env):
        lead_id = env.vehicles.get_leader(self.veh_id)
        this_vel = env.vehicles.get_speed(self.veh_id)
        lead_vel = env.vehicles.get_speed(lead_id)

        dx = env.vehicles.get_headway(self.veh_id)
        dv_minus = min(lead_vel - this_vel, 0)

        dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus ** 2
        dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus ** 2
        dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus ** 2

        # compute the desired velocity
        if dx <= dx_1:
            v_cmd = 0
        elif dx <= dx_2:
            v_cmd = this_vel * (dx-dx_1) / (dx_2-dx_1)
        elif dx <= dx_3:
            v_cmd = this_vel + (self.v_des-this_vel) * (dx-dx_2) / (dx_3-dx_2)
        else:
            v_cmd = self.v_des

        # compute the acceleration from the desired velocity
        return min((v_cmd - this_vel) / env.sim_step, self.max_accel)


class PISaturation(BaseController):

    def __init__(self, veh_id, max_accel=1):
        """Inspired by Dan Work's... work:

		Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments
		https://arxiv.org/abs/1705.01693

        Parameters
        ----------
        veh_id: str
            unique vehicle identifier
        max_accel: float, optional
            maximum achievable acceleration by the vehicle (m/s^2)
        """
        controller_params = {"delay": 1.0, "max_deaccel": 15,
                             "noise": 0, "fail_safe": None}
        BaseController.__init__(self, veh_id, controller_params)

        # maximum achievable acceleration by the vehicle
        self.max_accel = max_accel

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
        lead_id = env.vehicles.get_leader(self.veh_id)
        lead_vel = env.vehicles.get_speed(lead_id)
        this_vel = env.vehicles.get_speed(self.veh_id)

        dx = env.vehicles.get_headway(self.veh_id)
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
        self.v_cmd = beta * (alpha * v_target + (1-alpha) * lead_vel) \
            + (1-beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)
