from flow.core.generator import Generator
import numpy as np

class TwoLaneStraightMergeGenerator(Generator):
    """
    Generator class for a zi toll. No parameters needed
    from net_params (the network is not parametrized)
    """
    def specify_nodes(self, net_params):
        """
        See parent class
        """
        nodes = [{"id": "1", "x": "0",   "y": "0"},  # pre-toll
                 {"id": "2", "x": "200", "y": "0", "type": "priority"},  # merge
                 {"id": "3", "x": "400", "y": "0"}]  # post-merge2
        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        edges = [{"id": "1", "from": "1", "to": "2", "length": "200",  #
                  "spreadType": "center", "numLanes": "2", "speed": "23"},
                 {"id": "2", "from": "2", "to": "3", "length": "200",  # DONE
                  "spreadType": "center", "numLanes": "1", "speed": "23"}]
        return edges

    def specify_connections(self, net_params):
        """
        See parent class
        """
        conn = [{"from": "1", "to": "2", "fromLane": "0", "toLane": "0"},
                {"from": "1", "to": "2", "fromLane": "1", "toLane": "0"}]
        return conn

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"1": ["1", "2"],
               "2": ["2"],}

        return rts
