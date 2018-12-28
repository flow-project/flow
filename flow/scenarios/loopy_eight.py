"""Contains the fabulous loopy eight mixed autonomy scenario class."""

from flow.core.params import InitialConfig
from flow.scenarios.base_scenario import Scenario
import numpy as np
from numpy import linspace, pi, sin, cos
from copy import deepcopy

ADDITIONAL_NET_PARAMS = {}
SCALING = 30
RADIUS = 1.5

class LoopyEightScenario(Scenario):
    """Scenario class for bottleneck simulations."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig()):

        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.nodes = dict()

        super().__init__(name, vehicles, net_params, initial_config)

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
        rt = np.sqrt(2)
        nodes = [{'id': 'n1', 'x': -1, 'y': 0},
                 {'id': 'n2', 'x': 0, 'y': 1},
                 {'id': 'n3', 'x': 0, 'y': -1},
                 {'id': 'n4', 'x': 1, 'y': 0},
                 {'id': 'n6', 'x': 3, 'y': 1},
                 {'id': 'n7', 'x': 3, 'y': -1},
                 {'id': 'n8', 'x': 3 + rt, 'y': 0},
                 {'id': 'n9', 'x': 3 + rt/2, 'y': -rt/2},
                 {'id': 'n10', 'x': 3 + rt/2, 'y': rt/2},
                 {'id': 'n11', 'x': 3+1.5*rt, 'y': rt/2},
                 {'id': 'n12', 'x': 3+1.5*rt, 'y': -rt/2},]

        for node in nodes:
            self.nodes[node['id']] = np.array([node['x'] * RADIUS * SCALING,
                                               node['y'] * RADIUS * SCALING])

        for node in nodes:
            node['x'] = str(node['x'] * RADIUS * SCALING)
            node['y'] = str(node['y'] * RADIUS * SCALING)

        return nodes

    def specify_edges(self, net_params):
        res = 40
        rt = np.sqrt(2)
        edges = [{'id': 'e1', 'from': 'n3', 'to': 'n1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(cos(t), sin(t))
                            for t in linspace(3*pi/2, pi, res)]},
                 {'id': 'e2', 'from': 'n1', 'to': 'n2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(cos(t), sin(t))
                            for t in linspace(pi, pi/2, res)]},
                 {'id': 'e3', 'from': 'n2', 'to': 'n4', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(cos(t), sin(t))
                            for t in linspace(pi/2, 0, res)]},
                 {'id': 'e4', 'from': 'n4', 'to': 'n3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(cos(t), sin(t))
                            for t in linspace(2*pi, 3*pi/2, res)]},
                 {'id': 'e5', 'from': 'n2', 'to': 'n6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e6', 'from': 'n7', 'to': 'n3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e7', 'from': 'n7', 'to': 'n6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3 + cos(t), sin(t))
                            for t in linspace(3*pi/2, pi/2, res)]},
                 {'id': 'e8', 'from': 'n6', 'to': 'n10', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3 + cos(t), sin(t))
                            for t in linspace(pi/2, pi/4, res)]},
                 {'id': 'e9', 'from': 'n10', 'to': 'n8', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e10', 'from': 'n8', 'to': 'n9', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e11', 'from': 'n9', 'to': 'n7', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3 + cos(t), sin(t))
                            for t in linspace(7*pi/4, 3*pi/2, res)]},
                 {'id': 'e12', 'from': 'n8', 'to': 'n12', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e13', 'from': 'n12', 'to': 'n11', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3+2*rt + cos(t), sin(t))
                            for t in linspace(5*pi/4, 11*pi/4, res)]},
                 {'id': 'e14', 'from': 'n11', 'to': 'n8', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'}
                ]

        num_edge = len(edges)
        for i in range(num_edge):
            edge = edges[i]
            oppo_edge = {'id': edge['id'] + '_op',
                         'from': edge['to'],
                         'to': edge['from'],
                         'length': edge['length'],
                         'numLanes': edge['numLanes'],
                         'type': edge['type']}
            if edge.get('shape') is not None:
                print(edge['id'])
                oppo_edge['shape'] = edge['shape'][::-1]
                print("reversing...")
            edges.append(oppo_edge)

        for edge in edges:
            edge['numLanes'] = str(edge['numLanes'])
            if 'shape' in edge:
                edge['length'] = sum(
                    [np.sqrt((edge['shape'][i][0] - edge['shape'][i+1][0])**2 +
                             (edge['shape'][i][1] - edge['shape'][i+1][1])**2)
                     * SCALING for i in range(len(edge['shape'])-1)])
                edge['length'] = str(edge['length'])
                edge['shape'] = ' '.join('%.2f,%.2f' % (x*RADIUS*SCALING,
                                                        y*RADIUS*SCALING)
                                         for x, y in edge['shape'])
            else:
                edge['length'] = str(np.linalg.norm(self.nodes[edge['to']] -
                                                    self.nodes[edge['from']]))

            # fix junction overlapping issue
            # junctions = {'e_8_b': 2}
            # if edge['id'] in junctions:
            #     edge['length'] = str(junctions[edge['id']])

        return edges

    def specify_connections(self, net_params):
        conn = []
        # connect lanes at bottlenecks
        num_lanes = 2
    #     edges_from_b = ['e_13', 'e_42', 'e_60']  # order matters
    #     edges_to_b = ['e_14', 'e_44', 'e_69']
    #     for e_from, e_to in zip(edges_from_b, edges_to_b):
    #         for i in range(num_lanes):
    #             conn += [{
    #                 'from': e_from,
    #                 'to': e_to,
    #                 'fromLane': str(i),
    #                 'toLane': str(int(np.floor(i / 2)))
    #             }]
    #     # connect lanes at roundabout (order matters)
    #     edges_from_r = ['e_66', 'e_66', 'e_7', 'e_7', 'e_9', 'e_9']
    #     edges_to_r = ['e_91', 'e_63', 'e_17', 'e_8_b', 'e_10', 'e_92']
    #     for r_from, r_to in zip(edges_from_r, edges_to_r):
    #         for i in range(num_lanes):
    #             conn += [{
    #                 'from': r_from,
    #                 'to': r_to,
    #                 'fromLane': str(i),
    #                 'toLane': str(i)
    #             }]
    #
    #     # split one lane to two lanes from e_68 to e_66
    #     conn += [{
    #         'from': 'e_68',
    #         'to': 'e_66',
    #         'fromLane': '0',
    #         'toLane': '0'
    #     }]
    #
        # remove undesired left-turn connections
        edges_from_u = ['e1_op', 'e1_op', 'e2', 'e2', 'e3_op', 'e4', 'e5',
                        'e5_op', 'e6', 'e6_op', 'e7_op', 'e7', 'e8_op',
                        'e8_op', 'e11', 'e11']
        edges_to_u = ['e4_op', 'e6_op', 'e3', 'e5', 'e2_op', 'e1', 'e8',
                      'e2_op', 'e1', 'e11_op', 'e11_op', 'e8', 'e5_op',
                      'e7_op', 'e7', 'e6']
        for u_from, u_to in zip(edges_from_u, edges_to_u):
            for i in range(num_lanes):
                conn += [{
                    'from': u_from,
                    'to': u_to,
                    'fromLane': str(i),
                    'toLane': str(i)
                }]

        return conn

    def specify_types(self, net_params):
        types = [{'id': 'edgeType', 'speed': repr(15)}]
        return types

    def specify_routes(self, net_params):
        rts = {'e1': ['e1'],
               'e2': ['e2'],
               'e3': ['e3'],
               'e4': ['e4'],
               'e5': ['e5'],
               'e6': ['e6'],
               'e7': ['e7'],
               'e8': ['e8'],
               'e9': ['e9'],
               'e10': ['e10'],
               'e11': ['e11'],
               'e12': ['e12'],
               'e13': ['e13'],
               'e14': ['e14']}

        rts_op = {'e1_op': ['e1_op'],
                  'e2_op': ['e2_op'],
                  'e3_op': ['e3_op'],
                  'e4_op': ['e4_op'],
                  'e5_op': ['e5_op'],
                  'e6_op': ['e6_op'],
                  'e7_op': ['e7_op'],
                  'e8_op': ['e8_op'],
                  'e9_op': ['e9_op'],
                  'e10_op': ['e10_op'],
                  'e11_op': ['e11_op'],
                  'e12_op': ['e12_op'],
                  'e13_op': ['e13_op'],
                  'e14_op': ['e14_op']}
        rts = {**rts, **rts_op}
        return rts
