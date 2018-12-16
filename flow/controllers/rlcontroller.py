"""Contains the RLController class."""

from flow.controllers.base_controller import BaseController


class RLController(BaseController):
    """RL Controller.

    Vehicles with this class specified will be stored in the list of the RL IDs
    in the Vehicles class.

    Usage:

        >>> from flow.core.params import Vehicles
        >>> vehicles = Vehicles()
        >>> vehicles.add(acceleration_controller=(RLController, {}))

    In order to collect the list of all RL vehicles in the next, run:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> rl_ids = env.k.vehicle.get_rl_ids()
    """

    def __init__(self, veh_id, sumo_cf_params, time_delay=0, fail_safe=None):
        """Instantiates an RL Controller.

        Attributes
        ----------
        veh_id: str
            Vehicle ID for SUMO identification
        fail_safe: str, optional
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        """
        BaseController.__init__(
            self,
            veh_id,
            sumo_cf_params,
            delay=time_delay,
            fail_safe=fail_safe)
