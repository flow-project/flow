import numpy as np
from flow.controllers.base_controller import BaseController

class IDMController(BaseController):
    def __init__(self, veh_id, v0=30, T=1, a=1, b=1.5, 
                 delta=4, s0=2, s1=0, time_delay=0.0, 
                 dt=0.1, noise=0, fail_safe=None, car_following_params=None):
        """Docstring eliminated here for brevity"""
        BaseController.__init__(self, veh_id, car_following_params,
                                delay=time_delay, fail_safe=fail_safe,
                                noise=noise)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.s1 = s1
        self.dt = dt

        
    ##### Below this is new code #####
    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # negative headways may be registered by sumo at intersections/
        # junctions. Setting them to 0 causes vehicles to not move; therefore,
        # we maintain these negative headways to let sumo control the dynamics
        # as it sees fit at these points.
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0,
                v * self.T + v*(v-lead_vel) / (2*np.sqrt(self.a*self.b)))

        return self.a * (1 - (v/self.v0)**self.delta - (s_star/h)**2)


from flow.controllers.base_lane_changing_controller import BaseLaneChangeController

class LaneZeroController(BaseLaneChangeController):
    """A lane-changing model used to move vehicles into lane 0."""

    ##### Below this is new code #####
    def get_lane_change_action(self, env):
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        if current_lane > 0:
            return -1
        else:
            return 0
