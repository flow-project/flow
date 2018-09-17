import traci.constants as tc
from collections import defaultdict

# DEFAULTS
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.8
SHOW_DETECTORS = True

"""
Ideal Usage:
routes = Routes()
route.add(["top, "left", "bottom", "right"])
route.add(["top, "left", "bottom", "somethignelse"], prob=0.5)

"""

""" ORIGINAL ROUTE
# rts = {"top": ["top", "left", "bottom", "right"],
#        "left": ["left", "bottom", "right", "top"],
#        "bottom": ["bottom", "right", "top", "left"],
#        "right": ["right", "top", "left", "bottom"],
#        "inflow_1": ["inflow_1", "merge_in_1", "right", "top", "left", "bottom"],
#        "merge_in_1": ["merge_in_1", "right", "top", "left", "bottom"],
#        "inflow_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
#        "merge_in_0": ["merge_in_0", "left", "merge_out_1", "outflow_1"],
#        "merge_out_1": ["merge_out_1", "outflow_1"],
#        "outflow_1": ["outflow_1"],
#        }
"""
""" DISTRIBUTION ROUTE
rts =  {"top": {"top": ["top", "left", "bottom", "right"]},
        "left": {"left": ["left", "bottom", "right", "top"]},
        "bottom": {"bottom": ["bottom", "right", "top", "left"]},
        "right": {"right": ["right", "top", "left", "bottom"]},

        "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
                    "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
        "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
                    "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},
        "outflow_1": {"outflow_1": ["outflow_1"]},
        "outflow_0": {"outflow_0": ["outflow_0"]}
        }


This is the abstract of how I want my data kept:
routes = {"start_edge": [route, route]}

"""

class Routes:

    def __init__(self):#, distributed=False):
        """Base route.

        description here

        Parameters
        ----------
        baseline: bool
        """
        self.routes = {} # mapping of route_id -> route object
        self.starts = []

    def add(self, route_id, route, prob=1):
        """Adds a route to the network.

        for the base route case 
        Requires a unique route_id name
        Overwrites if an id with the same route_id is added

        Parameters
        ----------
        node_id : str
            name of the node with traffic lights
        tls_type : str, optional
            type of the traffic light (see Note)

        Note
        ----
        For information on defining traffic light properties, see:
        http://sumo.dlr.de/wiki/Simulation/Traffic_Lights#Defining_New_TLS-Programs
        """
        if route_id in self.routes.keys():
            print("Warning: Overwriting route with id %s" % route_id)
        start = route[0]
        if start not in self.starts:
            self.starts.append(start)
        route = {"route_id": route_id,
                 "route": route,
                 "start": start,
                 "prob": prob}
        self.routes[route_id] = route

    def get_distribution(self, start):
        """ 
        TODO Could maybe change this name
        """
        # return self.distribution[start]
        distrib = [r for r in self.routes.values() if r["start"]==start]
        return distrib

    def get_routes(self):
        """
        Return dictionary of all routes in the class.
        """
        return self.routes

    def get_starts(self):
        return self.starts


    def generate_routes(self):
        """
        To be used by the generator.
        """
        routes = {}
        for start in self.starts:
            routes[start] = self.get_distribution(start)
        return routes

    def remove(self, route_id):
        """
        Remove route with name ROUTE_ID.
        """
        del self.routes[route_id]
        # TODO clean up self.starts if necessary

    def get_ids(self):
        """Returns the names of all routes."""
        return self.routes.keys()
