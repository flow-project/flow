from flow.controllers.base_controller import BaseController


class RLController(BaseController):

    def __init__(self, veh_id, max_deacc=15, tau=0, dt=0.1, fail_safe=None):
        """Instantiates an RL Controller.

        Vehicles with this controller are provided with actions by an rl agent,
        and perform their actions accordingly.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        acc_max: float, optional
            max acceleration (default: 15)
        tau: float, optional
            time delay (default: 0)
        dt: float, optional
            timestep (default: 0.1)
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        controller_params = {"delay": tau/dt, "max_deaccel": max_deacc,
                             "fail_safe": fail_safe}
        BaseController.__init__(self, veh_id, controller_params)
