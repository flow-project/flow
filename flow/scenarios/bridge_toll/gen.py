from flow.core.generator import Generator
import numpy as np

class BBTollGenerator(Generator):
    """
    Generator class for simulating the Bay Bridge toll. No parameters needed
    from net_params (the network is not parametrized)
    """
    def specify_nodes(self, net_params):
        """
        See parent class
        """
        nodes = [{"id": "1", "x": "0",   "y": "0"},  # pre-toll
                 {"id": "2", "x": "100", "y": "0"},  # toll
                 {"id": "3", "x": "375", "y": "0"},  # light
                 {"id": "4", "x": "405", "y": "0", "type": "zipper"},  # merge1
                 {"id": "5", "x": "545", "y": "0", "type": "zipper"},  # merge2
                 {"id": "6", "x": "1500", "y": "0"}]  # post-merge2

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        scaling = net_params.additional_params.get("scaling", 1)
        assert(isinstance(scaling, int)), "Scaling must be an int"

        edges = [{"id": "1", "from": "1", "to": "2", "length": "100",  #
                  "spreadType": "center", "numLanes": str(4*scaling), "speed": "5"},
                 {"id": "2", "from": "2", "to": "3", "length": "275",  # DONE
                  "spreadType": "center", "numLanes": str(4*scaling), "speed": "5"},
                 {"id": "3", "from": "3", "to": "4", "length": "30",  # DONE
                  "spreadType": "center", "numLanes": str(4*scaling), "speed": "50"},
                 {"id": "4", "from": "4", "to": "5", "length": "140",   # DONE
                  "spreadType": "center", "numLanes": str(2*scaling), "speed": "50"},
                 {"id": "5", "from": "5", "to": "6", "length": "225",
                  "spreadType": "center", "numLanes": str(scaling), "speed": "50"}]

        return edges

    def specify_connections(self, net_params):
        """
        See parent class
        """
        scaling = net_params.additional_params.get("scaling", 1)
        conn = []
        {"from": "3", "to": "4", "fromLane": "0", "toLane": "0"},
        {"from": "3", "to": "4", "fromLane": "1", "toLane": "0"},
        {"from": "3", "to": "4", "fromLane": "2", "toLane": "1"},
        {"from": "3", "to": "4", "fromLane": "3", "toLane": "1"},
        for i in range(4*scaling):
            conn += [{"from": "3", "to": "4", "fromLane": str(i), "toLane": str(int(np.floor(i/2)))}]
        for i in range(2*scaling):
            conn += [{"from": "4", "to": "5", "fromLane": str(i), "toLane": str(int(np.floor(i/2)))}]

        # conn = [{"from": "3", "to": "4", "fromLane": "0", "toLane": "0"},
        #         {"from": "3", "to": "4", "fromLane": "1", "toLane": "0"},
        #         {"from": "3", "to": "4", "fromLane": "2", "toLane": "1"},
        #         {"from": "3", "to": "4", "fromLane": "3", "toLane": "1"},
        #         {"from": "3", "to": "4", "fromLane": "4", "toLane": "2"},
        #         {"from": "3", "to": "4", "fromLane": "5", "toLane": "2"},
        #         {"from": "3", "to": "4", "fromLane": "6", "toLane": "3"},
        #         {"from": "3", "to": "4", "fromLane": "7", "toLane": "3"},
        #         {"from": "3", "to": "4", "fromLane": "8", "toLane": "4"},
        #         {"from": "3", "to": "4", "fromLane": "9", "toLane": "4"},
        #         {"from": "3", "to": "4", "fromLane": "10", "toLane": "5"},
        #         {"from": "3", "to": "4", "fromLane": "11", "toLane": "5"},
        #         {"from": "3", "to": "4", "fromLane": "12", "toLane": "6"},
        #         {"from": "3", "to": "4", "fromLane": "13", "toLane": "6"},
        #         {"from": "3", "to": "4", "fromLane": "14", "toLane": "7"},
        #         {"from": "3", "to": "4", "fromLane": "15", "toLane": "7"},
        #         {"from": "4", "to": "5", "fromLane": "0", "toLane": "0"},
        #         {"from": "4", "to": "5", "fromLane": "1", "toLane": "1"},
        #         {"from": "4", "to": "5", "fromLane": "2", "toLane": "1"},
        #         {"from": "4", "to": "5", "fromLane": "3", "toLane": "2"},
        #         {"from": "4", "to": "5", "fromLane": "4", "toLane": "2"},
        #         {"from": "4", "to": "5", "fromLane": "5", "toLane": "3"},
        #         {"from": "4", "to": "5", "fromLane": "6", "toLane": "3"},
        #         {"from": "4", "to": "5", "fromLane": "7", "toLane": "4"}]
        # return conn
        # conn = [{"from": "3", "to": "4", "fromLane": "0", "toLane": "0"},
        #         {"from": "3", "to": "4", "fromLane": "1", "toLane": "0"},
        #         {"from": "3", "to": "4", "fromLane": "2", "toLane": "1"},
        #         {"from": "3", "to": "4", "fromLane": "3", "toLane": "1"},
        #         {"from": "4", "to": "5", "fromLane": "0", "toLane": "0"},
        #         {"from": "4", "to": "5", "fromLane": "1", "toLane": "0"},]
        return conn

    def specify_routes(self, net_params):
        """
        See parent class
        """
        rts = {"1": ["1", "2", "3", "4", "5"],
               "2": ["2", "3", "4", "5"],
               "3": ["3", "4", "5"],
               "4": ["4", "5"],
               "5": ["5"]}

        return rts
