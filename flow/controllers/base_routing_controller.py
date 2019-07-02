"""Contains the base routing controller class."""


class BaseRouter:
    """Base class for routing controllers.

    These controllers are used to dynamically change the routes of vehicles
    after initialization.

    Usage: 

        >>> from flow.core.vehicles import Vehicles
        >>> from flow.controllers import ContinuousRouter
        >>> vehicles = Vehicles()
        >>> veh_id = "human"
        >>> vehicles.add(veh_id, routing_controller=(ContinuousRouter, {}))

    You can replace "ContinuousRouter" with any routing controller you want. In order 
    to collect the list of all vehicles in the next, run:

        >>> veh_ids = vehicles.get_ids()

    Parameters
    ----------
    veh_id : str
        ID of the vehicle this controller is used for
    router_params : dict
        Dictionary of router params
    """

    def __init__(self, veh_id, router_params):
        """Instantiate the base class for routing controllers."""
        self.veh_id = veh_id
        self.router_params = router_params

    def choose_route(self, env):
        """Return the routing method implemented by the controller.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base_env.py

        Returns
        -------
        list or None
            The sequence of edges the vehicle should adopt. If a None value
            is returned, the vehicle performs no routing action in the current
            time step.
        """
        raise NotImplementedError
