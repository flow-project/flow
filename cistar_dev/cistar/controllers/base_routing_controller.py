
class BaseRouter:
    def __init__(self, veh_id, router_params):
        self.veh_id = veh_id
        self.router_params = router_params

    def choose_route(self, env):
        """
        The routing method implemented by the algorithm.

        :return: The sequence of edges the vehicle should adopt.
        If a None value is return, the vehicle performs no routing action
        in the current time step.
        """
        raise NotImplementedError
