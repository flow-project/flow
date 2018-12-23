from flow.controllers.base_controller import BaseController
import numpy as np


class FollowerStopper(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_des=15,
                 danger_edges=None):
        """Inspired by Dan Work's... work:

        Dissipation of stop-and-go waves via control of autonomous vehicles:
        Field experiments https://arxiv.org/abs/1705.01693

        Parameters
        ----------
        veh_id: str
            unique vehicle identifier
        v_des: float, optional
            desired speed of the vehicles (m/s)
        """
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
        env: Environment type
            see flow/envs/base_env.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.vehicles.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.scenario.edge_length(edge_id)
        relative_pos = env.vehicles.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.vehicles.get_leader(self.veh_id)
        this_vel = env.vehicles.get_speed(self.veh_id)
        lead_vel = env.vehicles.get_speed(lead_id)

        if self.v_des is None:
            return None

        if lead_id is None:
            v_cmd = self.v_des
        else:
            dx = env.vehicles.get_headway(self.veh_id)
            dv_minus = min(lead_vel - this_vel, 0)

            dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
            dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
            dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2

            # compute the desired velocity
            if dx <= dx_1:
                v_cmd = 0
            elif dx <= dx_2:
                v_cmd = this_vel * (dx - dx_1) / (dx_2 - dx_1)
            elif dx <= dx_3:
                v_cmd = this_vel + (self.v_des - this_vel) * (dx - dx_2) \
                        / (dx_3 - dx_2)
            else:
                v_cmd = self.v_des

        edge = env.vehicles.get_edge(self.veh_id)

        if edge == "":
            return None

        if self.find_intersection_dist(env) <= 10 and \
                env.vehicles.get_edge(self.veh_id) in self.danger_edges or \
                env.vehicles.get_edge(self.veh_id)[0] == ":":
            return None
        else:
            # compute the acceleration from the desired velocity
            return (v_cmd - this_vel) / env.sim_step


class PISaturation(BaseController):
    def __init__(self, veh_id, car_following_params):
        """Inspired by Dan Work's... work:

        Dissipation of stop-and-go waves via control of autonomous vehicles:
        Field experiments https://arxiv.org/abs/1705.01693

        Parameters
        ----------
        veh_id : str
            unique vehicle identifier
        car_following_params : SumoCarFollowingParams
            object defining sumo-specific car-following parameters
        """
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
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)


class HandTunedVelocityController(FollowerStopper):
    def __init__(self,
                 veh_id,
                 v_regions,
                 car_following_params,
                 danger_edges=None):
        super().__init__(
            veh_id, car_following_params, v_regions[0],
            danger_edges=danger_edges)
        self.v_regions = v_regions

    def get_accel(self, env):
        edge = env.vehicles.get_edge(self.veh_id)
        if edge:
            if edge[0] != ':' and edge in env.controlled_edges:
                pos = env.vehicles.get_position(self.veh_id)
                # find what segment we fall into
                bucket = np.searchsorted(env.slices[edge], pos) - 1
                action = self.v_regions[bucket +
                                        env.action_index[int(edge) - 1]]
                # set the desired velocity of the controller to the action
                controller = env.vehicles.get_acc_controller(self.veh_id)
                controller.v_des = action

        return super().get_accel(env)


class FeedbackController(FollowerStopper):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 Kp,
                 desired_bottleneck_density,
                 danger_edges=None):
        super().__init__(veh_id, car_following_params,
                         danger_edges=danger_edges)
        self.Kp = Kp
        self.desired_density = desired_bottleneck_density

    def get_accel(self, env):
        """See parent class."""
        current_lane = env.vehicles.get_lane(veh_id=self.veh_id)
        future_lanes = env.scenario.get_bottleneck_lanes(current_lane)
        future_edge_lanes = [
            "3_" + str(current_lane), "4_" + str(future_lanes[0]),
            "5_" + str(future_lanes[1])
        ]

        current_density = env.get_bottleneck_density(future_edge_lanes)
        edge = env.vehicles.get_edge(self.veh_id)
        if edge:
            if edge[0] != ':' and edge in env.controlled_edges:
                if edge in self.danger_edges:
                    self.v_des = None
                else:
                    self.v_des = max(
                        min(
                            self.v_des +
                            self.Kp * (self.desired_density - current_density),
                            23), 0)

        # print(current_density, self.v_des)
        return super().get_accel(env)
