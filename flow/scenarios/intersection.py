"""Contains the intersection scenario class."""

from flow.core.params import InitialConfig
from flow.scenarios.base_scenario import Scenario
import numpy as np


ADDITIONAL_NET_PARAMS = {}
SCALING = 10


class SoftIntersectionScenario(Scenario):
    """Scenario class for soft intersection simulations."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig()):
        """Instantiate the scenario class.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.nodes = dict()
        if net_params.junction_type is not None:
            self.junction_type = net_params.junction_type
        else:
            self.junction_type = 'priority'

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
        nodes = [
            {'id': 'n_1_inflow', 'x': -12, 'y': 0},
            {'id': 'n_1_sbc', 'x': -10, 'y': 0},
            {'id': 'n_1', 'x': -4, 'y': 0},
            {'id': 'n_2_inflow', 'x': 0, 'y': 12},
            {'id': 'n_2_sbc', 'x': 0, 'y': 10},
            {'id': 'n_2', 'x': 0, 'y': 4},
            {'id': 'n_3_inflow', 'x': 12, 'y': 0},
            {'id': 'n_3_sbc', 'x': 10, 'y': 0},
            {'id': 'n_3', 'x': 4, 'y': 0},
            {'id': 'n_4_inflow', 'x': 0, 'y': -12},
            {'id': 'n_4_sbc', 'x': 0, 'y': -10},
            {'id': 'n_4', 'x': 0, 'y': -4},
            {'id': 'n_5', 'x': 0, 'y': 0, 'type': self.junction_type}
        ]

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

        edges = [
            {'id': 'e_1_inflow', 'from': 'n_1_inflow', 'to': 'n_1_sbc', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_sbc+', 'from': 'n_1_sbc', 'to': 'n_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1', 'from': 'n_1', 'to': 'n_5', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2', 'from': 'n_5', 'to': 'n_1', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_sbc-', 'from': 'n_1', 'to': 'n_1_inflow', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_3_inflow', 'from': 'n_2_inflow', 'to': 'n_2_sbc', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_sbc+', 'from': 'n_2_sbc', 'to': 'n_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3', 'from': 'n_2', 'to': 'n_5', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4', 'from': 'n_5', 'to': 'n_2', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_sbc-', 'from': 'n_2', 'to': 'n_2_inflow', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_5_inflow', 'from': 'n_3_inflow', 'to': 'n_3_sbc', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_5_sbc+', 'from': 'n_3_sbc', 'to': 'n_3', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_5', 'from': 'n_3', 'to': 'n_5', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_6', 'from': 'n_5', 'to': 'n_3', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_6_sbc-', 'from': 'n_3', 'to': 'n_3_inflow', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_7_inflow', 'from': 'n_4_inflow', 'to': 'n_4_sbc', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_7_sbc+', 'from': 'n_4_sbc', 'to': 'n_4', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_7', 'from': 'n_4', 'to': 'n_5', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_8', 'from': 'n_5', 'to': 'n_4', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_8_sbc-', 'from': 'n_4', 'to': 'n_4_inflow', 'length': None,
                'numLanes': 2, 'type': 'edgeType'},
        ]

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
        # speed limit 25 mph
        types = [{'id': 'edgeType', 'speed': repr(11.176)}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {'e_1_sbc+': ['e_1_sbc+'],
               'e_2_sbc-': ['e_2_sbc-'],
               'e_3_sbc+': ['e_3_sbc+'],
               'e_4_sbc-': ['e_4_sbc-'],
               'e_5_sbc+': ['e_5_sbc+'],
               'e_6_sbc-': ['e_6_sbc-'],
               'e_7_sbc+': ['e_7_sbc+'],
               'e_8_sbc-': ['e_8_sbc-'],
               'e_1': ['e_1'],
               'e_2': ['e_2'],
               'e_3': ['e_3'],
               'e_4': ['e_4'],
               'e_5': ['e_5'],
               'e_6': ['e_6'],
               'e_7': ['e_7'],
               'e_8': ['e_8'],
               'e_1_inflow': ['e_1_inflow'],
               'e_3_inflow': ['e_3_inflow'],
               'e_5_inflow': ['e_5_inflow'],
               'e_7_inflow': ['e_7_inflow'],}
        return rts

class HardIntersectionScenario(SoftIntersectionScenario):
    """Scenario class for hard intersection simulations."""

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [
            {'id': 'n_1_zone1', 'x': -10, 'y': 0},
            {'id': 'n_1_zone2', 'x': -7.5, 'y': 0},
            {'id': 'n_1_zone3', 'x': -5, 'y': 0},
            {'id': 'n_1_zone4', 'x': -2.5, 'y': 0},
            {'id': 'n_2_zone1', 'x': 0, 'y': 10},
            {'id': 'n_2_zone2', 'x': 0, 'y': 7.5},
            {'id': 'n_2_zone3', 'x': 0, 'y': 5},
            {'id': 'n_2_zone4', 'x': 0, 'y': 2.5},
            {'id': 'n_3_zone1', 'x': 10, 'y': 0},
            {'id': 'n_3_zone2', 'x': 7.5, 'y': 0},
            {'id': 'n_3_zone3', 'x': 5, 'y': 0},
            {'id': 'n_3_zone4', 'x': 2.5, 'y': 0},
            {'id': 'n_4_zone1', 'x': 0, 'y': -10},
            {'id': 'n_4_zone2', 'x': 0, 'y': -7.5},
            {'id': 'n_4_zone3', 'x': 0, 'y': -5},
            {'id': 'n_4_zone4', 'x': 0, 'y': -2.5},
            {'id': 'n_5', 'x': 0, 'y': 0, 'type': self.junction_type}]

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

        edges = [
            {'id': 'e_1_zone1+', 'from': 'n_1_zone1', 'to': 'n_1_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone2+', 'from': 'n_1_zone2', 'to': 'n_1_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone3+', 'from': 'n_1_zone3', 'to': 'n_1_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone4+', 'from': 'n_1_zone4', 'to': 'n_5',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_1_zone1-', 'from': 'n_1_zone2', 'to': 'n_1_zone1',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone2-', 'from': 'n_1_zone3', 'to': 'n_1_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone3-', 'from': 'n_1_zone4', 'to': 'n_1_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_1_zone4-', 'from': 'n_5', 'to': 'n_1_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_2_zone1+', 'from': 'n_2_zone1', 'to': 'n_2_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone2+', 'from': 'n_2_zone2', 'to': 'n_2_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone3+', 'from': 'n_2_zone3', 'to': 'n_2_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone4+', 'from': 'n_2_zone4', 'to': 'n_5',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_2_zone1-', 'from': 'n_2_zone2', 'to': 'n_2_zone1',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone2-', 'from': 'n_2_zone3', 'to': 'n_2_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone3-', 'from': 'n_2_zone4', 'to': 'n_2_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_2_zone4-', 'from': 'n_5', 'to': 'n_2_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_3_zone1+', 'from': 'n_3_zone1', 'to': 'n_3_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone2+', 'from': 'n_3_zone2', 'to': 'n_3_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone3+', 'from': 'n_3_zone3', 'to': 'n_3_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone4+', 'from': 'n_3_zone4', 'to': 'n_5',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_3_zone1-', 'from': 'n_3_zone2', 'to': 'n_3_zone1',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone2-', 'from': 'n_3_zone3', 'to': 'n_3_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone3-', 'from': 'n_3_zone4', 'to': 'n_3_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_3_zone4-', 'from': 'n_5', 'to': 'n_3_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_4_zone1+', 'from': 'n_4_zone1', 'to': 'n_4_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone2+', 'from': 'n_4_zone2', 'to': 'n_4_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone3+', 'from': 'n_4_zone3', 'to': 'n_4_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone4+', 'from': 'n_4_zone4', 'to': 'n_5',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},

            {'id': 'e_4_zone1-', 'from': 'n_4_zone2', 'to': 'n_4_zone1',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone2-', 'from': 'n_4_zone3', 'to': 'n_4_zone2',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone3-', 'from': 'n_4_zone4', 'to': 'n_4_zone3',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
            {'id': 'e_4_zone4-', 'from': 'n_5', 'to': 'n_4_zone4',
                'length': None, 'numLanes': 2, 'type': 'edgeType'},
        ]

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

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            'e_1_zone1+': ['e_1_zone1+', 'e_1_zone2+',
                          'e_1_zone3+', 'e_1_zone4+',
                          'e_3_zone4-', 'e_3_zone3-',
                          'e_3_zone2-', 'e_3_zone1-',],
            'e_1_zone2+': ['e_1_zone2+',
                          'e_1_zone3+', 'e_1_zone4+',
                          'e_3_zone4-', 'e_3_zone3-',
                          'e_3_zone2-', 'e_3_zone1-',],
            'e_1_zone3+': ['e_1_zone3+', 'e_1_zone4+',
                          'e_3_zone4-', 'e_3_zone3-',
                          'e_3_zone2-', 'e_3_zone1-',],
            'e_1_zone1-': ['e_1_zone4-', 'e_1_zone3-',
                          'e_1_zone2-', 'e_1_zone1-',],

            'e_2_zone1+': ['e_2_zone1+', 'e_2_zone2+',
                          'e_2_zone3+', 'e_2_zone4+',
                          'e_4_zone4-', 'e_4_zone3-',
                          'e_4_zone2-', 'e_4_zone1-',],
            'e_2_zone2+': ['e_2_zone2+',
                          'e_2_zone3+', 'e_2_zone4+',
                          'e_4_zone4-', 'e_4_zone3-',
                          'e_4_zone2-', 'e_4_zone1-',],
            'e_2_zone3+': ['e_2_zone3+', 'e_2_zone4+',
                          'e_4_zone4-', 'e_4_zone3-',
                          'e_4_zone2-', 'e_4_zone1-',],
            'e_2_zone1-': ['e_2_zone4-', 'e_2_zone3-',
                          'e_2_zone2-', 'e_2_zone1-',],

            'e_3_zone1+': ['e_3_zone1+', 'e_3_zone2+',
                          'e_3_zone3+', 'e_3_zone4+',
                          'e_1_zone4-', 'e_1_zone3-',
                          'e_1_zone2-', 'e_1_zone1-',],
            'e_3_zone2+': ['e_3_zone2+',
                          'e_3_zone3+', 'e_3_zone4+',
                          'e_1_zone4-', 'e_1_zone3-',
                          'e_1_zone2-', 'e_1_zone1-',],
            'e_3_zone3+': ['e_3_zone3+', 'e_3_zone4+',
                          'e_1_zone4-', 'e_1_zone3-',
                          'e_1_zone2-', 'e_1_zone1-',],
            'e_3_zone1-': ['e_3_zone4-', 'e_3_zone3-',
                          'e_3_zone2-', 'e_3_zone1-',],

            'e_4_zone1+': ['e_4_zone1+', 'e_4_zone2+',
                          'e_4_zone3+', 'e_4_zone4+',
                          'e_2_zone4-', 'e_2_zone3-',
                          'e_2_zone2-', 'e_2_zone1-',],
            'e_4_zone2+': ['e_4_zone2+',
                          'e_4_zone3+', 'e_4_zone4+',
                          'e_2_zone4-', 'e_2_zone3-',
                          'e_2_zone2-', 'e_2_zone1-',],
            'e_4_zone3+': ['e_4_zone3+', 'e_4_zone4+',
                          'e_2_zone4-', 'e_2_zone3-',
                          'e_2_zone2-', 'e_2_zone1-',],
            'e_4_zone1-': ['e_4_zone4-', 'e_4_zone3-',
                          'e_4_zone2-', 'e_4_zone1-',],
            }

        return rts
