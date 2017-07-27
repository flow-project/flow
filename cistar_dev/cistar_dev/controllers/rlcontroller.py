
from cistar_dev.controllers.base_controller import BaseController
from cistar_dev.controllers.base_lane_changing_controller import BaseLaneChangingController


class RLController(BaseController):
    """ Base RL Controller (assumes acceleration by Default)
    """

    def __init__(self, veh_id, max_deacc=15, tau=0, dt=0.1):
        """Instantiates a CFM controller

        Arguments:
            veh_id -- Vehicle ID for SUMO identification

        Keyword Arguments:
            acc_max {number} -- [max acceleration] (default: {15})
            tau {number} -- [time delay] (default: {0})
            dt {number} -- [timestep] (default: {0.1})
        """

        controller_params = {"delay": tau/dt, "max_deaccel": max_deacc}
        BaseController.__init__(self, veh_id, controller_params)


class RLLaneChangeController(BaseLaneChangingController):
    """ Base RL Lane-Changing Controller
    """

    def __init__(self, veh_id, lane_change_params=None):
        """

        :param veh_id:
        :param controller_params:
        """
        if lane_change_params is None:
            lane_change_params = {}

        super().__init__(veh_id, lane_change_params)
