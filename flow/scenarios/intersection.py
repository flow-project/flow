"""Contains the intersection scenario class."""

from flow.core.params import InitialConfig
from flow.scenarios.base_scenario import Scenario
import numpy as np


ADDITIONAL_NET_PARAMS = {}
SCALING = 10


class IntersectionScenario(Scenario):
    """Scenario class for bottleneck simulations."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig()):
        """Instantiate the scenario class.

        Requires from net_params:
        - scaling: the factor multiplying number of lanes

        In order for right-of-way dynamics to take place at the intersection,
        set 'no_internal_links' in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.nodes = dict()

        super().__init__(name, vehicles, net_params,initial_config)

    def specify_edge_starts(self):
        """See parent class."""
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in self._edge_list:
            # the current edge starts where the last edge ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self._edges[edge_id]['length']

        return edgestarts

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [{'id': 'n_1', 'x': -10, 'y': 0},
                 {'id': 'n_2', 'x': 0, 'y': 10},
                 {'id': 'n_3', 'x': 10, 'y': 0},
                 {'id': 'n_4', 'x': 0, 'y': -10},
                 {'id': 'n_5', 'x': 0, 'y': 0}]

        for node in nodes:
            self.nodes[node['id']] = np.array([node['x'] * SCALING,
                                               node['y'] * SCALING])

        for node in nodes:
            node['x'] = str(node['x'] * SCALING)
            node['y'] = str(node['y'] * SCALING)

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        res = 40

        edges = [{'id': 'e_1', 'from': 'n_1', 'to': 'n_5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_2', 'from': 'n_5', 'to': 'n_1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_3', 'from': 'n_2', 'to': 'n_5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_4', 'from': 'n_5', 'to': 'n_2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_5', 'from': 'n_3', 'to': 'n_5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_6', 'from': 'n_5', 'to': 'n_3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_7', 'from': 'n_4', 'to': 'n_5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_8', 'from': 'n_5', 'to': 'n_4', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'}]

        for edge in edges:
            edge['numLanes'] = str(edge['numLanes'])
            if 'shape' in edge:
                edge['length'] = sum(
                    [np.sqrt((edge['shape'][i][0] - edge['shape'][i+1][0])**2 +
                             (edge['shape'][i][1] - edge['shape'][i+1][1])**2)
                     * SCALING for i in range(len(edge['shape'])-1)])
                edge['length'] = str(edge['length'])
                edge['shape'] = ' '.join('%.2f,%.2f' % (blip*SCALING,
                                                        blop*SCALING)
                                         for blip, blop in edge['shape'])
            else:
                edge['length'] = str(np.linalg.norm(self.nodes[edge['to']] -
                                                    self.nodes[edge['from']]))

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        types = [{'id': 'edgeType', 'speed': repr(12)}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {'e_1': ['e_1', 'e_6'],
               'e_2': ['e_2'],
               'e_3': ['e_3', 'e_8'],
               'e_4': ['e_4'],
               'e_5': ['e_5', 'e_2'],
               'e_6': ['e_6'],
               'e_7': ['e_7', 'e_4'],
               'e_8': ['e_8']}

        return rts
