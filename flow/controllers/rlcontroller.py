"""Contains the RLController class."""

from flow.controllers.base_controller import BaseController


class RLController(BaseController):
    """RL Controller.

    Vehicles with this class specified will be stored in the list of the RL IDs
    in the Vehicles class.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification

    Examples
    --------
    A set of vehicles can be instantiated as RL vehicles as follows:

        >>> from flow.core.params import VehicleParams
        >>> vehicles = VehicleParams()
        >>> vehicles.add(acceleration_controller=(RLController, {}))

    In order to collect the list of all RL vehicles in the next, run:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> rl_ids = env.k.vehicle.get_rl_ids()
    """

    def __init__(self, veh_id, car_following_params):
        """Instantiate an RL Controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params)
