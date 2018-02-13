
from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """
    A router used to maintain continuous re-routing of the vehicle. This class
    is useful if vehicles are expected to continuously follow the same route,
    and repeat said route once it reaches its end.
    """
    def choose_route(self, env):
        """
        See parent class
        """
        if env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            new_route = env.available_routes[env.vehicles.get_edge(self.veh_id)]
        else:
            new_route = None

        return new_route


class BayBridgeRouter(ContinuousRouter):
    """
    Extension to the Continuous Router. Assists in choosing routes in select
    cases.
    """
    def choose_route(self, env):
        """
        See parent class
        """
        edge = env.vehicles.get_edge(self.veh_id)
        lane = env.vehicles.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"]
        else:
            new_route = super().choose_route(env)

        return new_route
