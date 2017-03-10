from cistar.controllers.base_controller import BaseController


class ConstantVelocityController(BaseController):
    """Base velocity controller (assumes acceleration by Default)
    """

    def __init__(self, veh_id, deacc_max=15, tau=0, dt=0.1, constant_speed=15):
        """Instantiates a velocity controller

        Arguments:
            veh_id -- Vehicle ID for SUMO identification

        Keyword Arguments:
            deacc_max {number} -- [max deacceleration] (default: {15})
            tau {number} -- [time delay] (default: {0})
            dt {number} -- [timestep] (default: {0.1})
            constant_speed {number} -- [target constant velocity] (default: {15})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": deacc_max}
        BaseController.__init__(self, veh_id, controller_params)
        self.constant_speed = constant_speed

    def get_action(self, env):
        return self.constant_speed

    def get_safe_action(self, env, action):
        v_safe = self.safe_velocity(env)
        if v_safe < action:
            print(v_safe, action)
        return min(action, v_safe)

    def reset_delay(self, env):
        pass
