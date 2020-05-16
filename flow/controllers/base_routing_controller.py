"""Contains the base routing controller class."""

from abc import ABCMeta, abstractmethod


class BaseRouter(metaclass=ABCMeta):
    """Base class for routing controllers.

    These controllers are used to dynamically change the routes of vehicles
    after initialization.

    Usage
    -----
    >>> from flow.core.params import VehicleParams
    >>> from flow.controllers import ContinuousRouter
    >>> vehicles = VehicleParams()
    >>> vehicles.add("human", routing_controller=(ContinuousRouter, {}))

    Note: You can replace "ContinuousRouter" with any routing controller you
    want.

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

    @abstractmethod
    def choose_route(self, env):
        """Return the routing method implemented by the controller.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        list or None
            The sequence of edges the vehicle should adopt. If a None value
            is returned, the vehicle performs no routing action in the current
            time step.
        """
        pass
