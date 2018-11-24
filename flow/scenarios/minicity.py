"""Contains the bottleneck scenario class."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario
import numpy as np
from numpy import linspace, pi, sin, cos

ADDITIONAL_NET_PARAMS = {}
SCALING = 40


class MiniCityScenario(Scenario):
    """Scenario class for bottleneck simulations."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
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

        super().__init__(name, vehicles, net_params,
                         initial_config, traffic_lights)

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
        nodes = [{'id': 'n_i1', 'x': 0.65, 'y': 2.75},
                 {'id': 'n_i2', 'x': 0.65, 'y': 4.90},
                 {'id': 'n_i3', 'x': 3.05, 'y': 2.75},
                 {'id': 'n_i4', 'x': 3.05, 'y': 4.90},
                 {'id': 'n_i5', 'x': 3.05, 'y': 5.75},
                 {'id': 'n_i6', 'x': 5.08, 'y': 2.75},
                 {'id': 'n_i7', 'x': 5.08, 'y': 3.75},
                 {'id': 'n_i8', 'x': 5.08, 'y': 4.90},
                 {'id': 'n_i9', 'x': 5.08, 'y': 5.75},
                 {'id': 'n_r1', 'x': 0.30, 'y': 1.15},
                 {'id': 'n_r2', 'x': 0.90, 'y': 0.40},
                 {'id': 'n_r3', 'x': 0.95, 'y': 1.45},
                 {'id': 'n_r4', 'x': 1.30, 'y': 1.13},
                 {'id': 'n_r4_tmp', 'x': 1.83 - 0.53 * cos(0.2 * pi),
                  'y': 0.94 - 0.53 * sin(0.2 * pi)},
                 {'id': 'n_r5', 'x': 4.55, 'y': 1.15},
                 {'id': 'n_r6', 'x': 4.85, 'y': 1.60},
                 {'id': 'n_r7', 'x': 5.10, 'y': 0.64},
                 {'id': 'n_r8', 'x': 5.35, 'y': 1.60},
                 {'id': 'n_b1', 'x': 0.65, 'y': 3.70},
                 {'id': 'n_b2', 'x': 0.65, 'y': 4.20},
                 {'id': 'n_b3', 'x': 3.05, 'y': 3.75},
                 {'id': 'n_b4', 'x': 3.05, 'y': 4.20},
                 {'id': 'n_b5', 'x': 4.10, 'y': 2.75},
                 {'id': 'n_b6', 'x': 4.55, 'y': 2.75},
                 {'id': 'n_m1_u', 'x': 1.83, 'y': 0.40},
                 {'id': 'n_m1_b', 'x': 1.83, 'y': 0.40},
                 {'id': 'n_m2', 'x': 1.96, 'y': 2.75},
                 {'id': 'n_m3', 'x': 2.18, 'y': 4.90},
                 {'id': 'n_m4', 'x': 2.50, 'y': 0.40},
                 {'id': 'n_m5', 'x': 4.20, 'y': 4.90},
                 {'id': 'n_s1', 'x': 0.65, 'y': 1.80},
                 {'id': 'n_s2', 'x': 1.35, 'y': 4.07},
                 {'id': 'n_s3', 'x': 1.67, 'y': 4.52},
                 {'id': 'n_s4', 'x': 2.34, 'y': 1.14},
                 {'id': 'n_s5', 'x': 2.34, 'y': 1.98},
                 {'id': 'n_s6', 'x': 3.05, 'y': 1.15},
                 {'id': 'n_s7_l', 'x': 3.51, 'y': 0.40},
                 {'id': 'n_s7_r', 'x': 3.95, 'y': 0.40},
                 {'id': 'n_s8', 'x': 4.20, 'y': 4.30},
                 {'id': 'n_s9', 'x': 5.08, 'y': 2.22},
                 {'id': 'n_s10', 'x': 5.63, 'y': 4.30},
                 {'id': 'n_s11', 'x': 5.63, 'y': 5.21},
                 {'id': 'n_s12', 'x': 0.65, 'y': 5.25},
                 {'id': 'n_s13', 'x': 1.20, 'y': 5.75},
                 {'id': 'n_s14', 'x': 4.75, 'y': 3.75}]

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

        edges = [{'id': 'e_1', 'from': 'n_s1', 'to': 'n_r1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (-0.38 + 1.03 * cos(t), 1.80 + 1.03 * sin(t))
                      for t in linspace(2 * pi, 2 * pi - 0.56 / 1.03, res)]
                  },
                 {'id': 'e_2', 'from': 'n_i1', 'to': 'n_s1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_3', 'from': 'n_b1', 'to': 'n_i1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_4', 'from': 'n_b2', 'to': 'n_b1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_5', 'from': 'n_i2', 'to': 'n_b2', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_6', 'from': 'n_s12', 'to': 'n_i2', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_7', 'from': 'n_r1', 'to': 'n_r2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (0.90 + 0.53 * cos(t), 0.93 + 0.53 * sin(t))
                      for t in linspace(3*pi/2 - 1.16/0.53, 3*pi/2, res)]
                  },
                 {'id': 'e_8_b', 'from': 'n_r2', 'to': 'n_r4_tmp',
                  'length': None, 'numLanes': 2, 'type': 'edgeType',
                  'shape': [(0.90 + 0.53 * cos(t), 0.93 + 0.53 * sin(t))
                            for t in linspace(3 * pi / 2, 1.8 * pi, res)]
                  },
                 {'id': 'e_8_u', 'from': 'n_r4_tmp', 'to': 'n_r4',
                  'length': None, 'numLanes': 2, 'type': 'edgeType',
                  'shape': [(0.90 + 0.53 * cos(t), 0.93 + 0.53 * sin(t))
                            for t in linspace(0, 1.16 / 0.53 - pi / 2, res)]
                  },
                 {'id': 'e_9', 'from': 'n_r4', 'to': 'n_r3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (0.90 + 0.53 * cos(t), 0.93 + 0.53 * sin(t))
                      for t in linspace(3*pi/2 + 1.16/0.53, 3*pi/2 + 1.46/0.53,
                                        res)]
                  },
                 {'id': 'e_10', 'from': 'n_r3', 'to': 'n_s1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (1.08 + 0.43 * cos(t), 1.90 + 0.43 * sin(t))
                      for t in reversed(linspace(pi, pi + 0.50 / 0.43, res))]
                  },
                 {'id': 'e_11', 'from': 'n_s1', 'to': 'n_i1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_12', 'from': 'n_i1', 'to': 'n_b1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_13', 'from': 'n_b1', 'to': 'n_b2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_14', 'from': 'n_b2', 'to': 'n_i2', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_15', 'from': 'n_i2', 'to': 'n_s12', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_16', 'from': 'n_s12', 'to': 'n_s13', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.19 + 0.54 * cos(t), 5.21 + 0.54 * sin(t))
                            for t in reversed(linspace(pi / 2, pi, res))]
                  },
                 {'id': 'e_17', 'from': 'n_r2', 'to': 'n_m1_b', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_18', 'from': 'n_b1', 'to': 'n_s2', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.11 + 0.46 * cos(t), 3.61 + 0.46 * sin(t))
                            for t in reversed(linspace(0.65 / 0.46, pi, res))]
                  },
                 {'id': 'e_19', 'from': 'n_s2', 'to': 'n_s3', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.21 + 0.46 * cos(t), 4.53 + 0.46 * sin(t))
                            for t in linspace(2*pi - 0.78/0.46, 2*pi, res)]
                  },
                 {'id': 'e_20', 'from': 'n_s13', 'to': 'n_i5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_21', 'from': 'n_m1_u', 'to': 'n_r4_tmp',
                  'length': None, 'numLanes': 2, 'type': 'edgeType',
                  'shape': [(1.89 + 0.54 * cos(t), 0.94 + 0.54 * sin(t))
                            for t in reversed(linspace(1.25*pi, 3*pi/2, res))]
                  },
                 {'id': 'e_22', 'from': 'n_i2', 'to': 'n_m3', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_23', 'from': 'n_m3', 'to': 'n_i2', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_24', 'from': 'n_s3', 'to': 'n_m3', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(2.13 + 0.46 * cos(t), 4.44 + 0.46 * sin(t))
                            for t in reversed(linspace(pi / 2, pi, res))]
                  },
                 {'id': 'e_25', 'from': 'n_i1', 'to': 'n_m2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_26', 'from': 'n_m2', 'to': 'n_i1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_27', 'from': 'n_s13', 'to': 'n_s12', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.19 + 0.54 * cos(t), 5.21 + 0.54 * sin(t))
                            for t in linspace(pi / 2, pi, res)]
                  },
                 {'id': 'e_28_b', 'from': 'n_m1_b', 'to': 'n_m4',
                  'length': None, 'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_29_u', 'from': 'n_m4', 'to': 'n_m1_u',
                  'length': None, 'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_30', 'from': 'n_m2', 'to': 'n_s5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.88 + 0.46 * cos(t), 2.29 + 0.46 * sin(t))
                            for t in reversed(linspace(0, pi / 2, res))]
                  },
                 {'id': 'e_31', 'from': 'n_s5', 'to': 'n_s4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_32', 'from': 'n_s4', 'to': 'n_m1_u', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(1.88 + 0.46 * cos(t), 0.86 + 0.46 * sin(t))
                            for t in reversed(linspace(3*pi/2, 2*pi, res))]
                  },
                 {'id': 'e_33', 'from': 'n_m3', 'to': 'n_i4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_34', 'from': 'n_i4', 'to': 'n_m3', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_35', 'from': 'n_i5', 'to': 'n_s13', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_36', 'from': 'n_m4', 'to': 'n_s7_l', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_37', 'from': 'n_s6', 'to': 'n_m4', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(2.59 + 0.46 * cos(t), 0.86 + 0.46 * sin(t))
                            for t in reversed(linspace(3*pi/2, 2*pi, res))]
                  },
                 {'id': 'e_38', 'from': 'n_s6', 'to': 'n_i3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_39', 'from': 'n_i3', 'to': 'n_s6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_40', 'from': 'n_i3', 'to': 'n_b3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_41', 'from': 'n_b3', 'to': 'n_i3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_42', 'from': 'n_b3', 'to': 'n_b4', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_43', 'from': 'n_b4', 'to': 'n_b3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_44', 'from': 'n_b4', 'to': 'n_i4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_45', 'from': 'n_i4', 'to': 'n_b4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_46', 'from': 'n_i4', 'to': 'n_i5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_47', 'from': 'n_i5', 'to': 'n_i4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_48', 'from': 'n_i5', 'to': 'n_i9', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_49', 'from': 'n_i4', 'to': 'n_m5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_50', 'from': 'n_i3', 'to': 'n_b5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_51', 'from': 'n_s7_l', 'to': 'n_m4', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_52', 'from': 'n_s7_l', 'to': 'n_s6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3.51 + 0.46 * cos(t), 0.86 + 0.46 * sin(t))
                            for t in reversed(linspace(pi, 3 * pi / 2, res))]
                  },
                 {'id': 'e_53', 'from': 'n_s7_r', 'to': 'n_r7', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3.95 + 3.84 * cos(t), 4.39 + 3.84 * sin(t) - 0.15)
                            for t in linspace(3 * pi / 2 - 0.1 / 3.84,
                                              3 * pi / 2 + 1.3 / 3.84, res)]
                  },
                 {'id': 'e_54', 'from': 'n_b5', 'to': 'n_i3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_55', 'from': 'n_m5', 'to': 'n_s8', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_56', 'from': 'n_s8', 'to': 'n_s14', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(4.75 - 0.54 * cos(t), 4.29 - 0.54 * sin(t))
                            for t in linspace(0, pi / 2, res)]
                  },
                 {'id': 'e_57', 'from': 'n_s8', 'to': 'n_m5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_58', 'from': 'n_m5', 'to': 'n_i8', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_59', 'from': 'n_m5', 'to': 'n_i4', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_60', 'from': 'n_b5', 'to': 'n_b6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_61', 'from': 'n_b6', 'to': 'n_b5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_62', 'from': 'n_s14', 'to': 'n_s8', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(4.75 - 0.54 * sin(t), 4.29 - 0.54 * cos(t))
                            for t in linspace(0, pi / 2, res)]
                  },
                 {'id': 'e_63', 'from': 'n_r5', 'to': 'n_s7_r', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(3.95 + 0.63 * cos(t), 1.03 + 0.63 * sin(t))
                            for t in reversed(linspace(3*pi/2, 2*pi, res))]
                  },
                 {'id': 'e_64', 'from': 'n_r7', 'to': 'n_r8', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(5.08 + 0.58 * cos(t), 1.17 + 0.58 * sin(t))
                            for t in linspace(3*pi/2, pi + 2.58/0.58, res)]
                  },
                 {'id': 'e_65', 'from': 'n_r8', 'to': 'n_r6', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (5.08 + 0.58 * cos(t), 1.17 + 0.58 * sin(t))
                      for t in linspace(pi + 2.58 / 0.58,
                                        1.2 * pi + 2.58 / 0.58, res)]
                  },
                 {'id': 'e_66', 'from': 'n_r6', 'to': 'n_r5', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [
                      (5.08 + 0.58 * cos(t), 1.17 + 0.58 * sin(t))
                      for t in linspace(1.2 * pi + 2.58 / 0.58,
                                        1.66 * pi + 2.58 / 0.58, res)]
                  },
                 {'id': 'e_67', 'from': 'n_r8', 'to': 'n_s9', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [
                      (5.75 + 0.67 * cos(t), 2.23 + 0.67 * sin(t))
                      for t in reversed(linspace(pi, pi + 0.64/0.67, res))]
                  },
                 {'id': 'e_68', 'from': 'n_s9', 'to': 'n_r6', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [
                      (4.51 + 0.57 * cos(t), 2.27 + 0.57 * sin(t))
                      for t in reversed(linspace(- 0.42 / 0.57, 0, res))]
                  },
                 {'id': 'e_69', 'from': 'n_b6', 'to': 'n_i6', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_70', 'from': 'n_i6', 'to': 'n_b6', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_71', 'from': 'n_s9', 'to': 'n_i6', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_72', 'from': 'n_i6', 'to': 'n_s9', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_73', 'from': 'n_i6', 'to': 'n_i7', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_74', 'from': 'n_i7', 'to': 'n_i6', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_75', 'from': 'n_i7', 'to': 'n_i8', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_76', 'from': 'n_i8', 'to': 'n_i7', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_77', 'from': 'n_i8', 'to': 'n_i9', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_78', 'from': 'n_i9', 'to': 'n_i8', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_79', 'from': 'n_i9', 'to': 'n_i5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_80', 'from': 'n_i7', 'to': 'n_s10', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(5.08 + 0.54 * cos(t), 4.29 + 0.54 * sin(t))
                            for t in linspace(-pi / 2, 0, res)],
                  },
                 {'id': 'e_81', 'from': 'n_i9', 'to': 'n_s11', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(5.08 - 0.54 * cos(t), 5.21 + 0.54 * sin(t))
                            for t in linspace(pi / 2, pi, res)]
                  },
                 {'id': 'e_82', 'from': 'n_s11', 'to': 'n_i9', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(5.08 + 0.54 * cos(t), 5.21 + 0.54 * sin(t))
                            for t in linspace(0, pi / 2, res)]
                  },
                 {'id': 'e_83', 'from': 'n_s10', 'to': 'n_s11', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_84', 'from': 'n_s11', 'to': 'n_s10', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_85', 'from': 'n_s10', 'to': 'n_i7', 'length': None,
                  'numLanes': 1, 'type': 'edgeType',
                  'shape': [(5.08 + 0.54 * cos(t), 4.29 - 0.54 * sin(t))
                            for t in linspace(0, pi / 2, res)]
                  },
                 {'id': 'e_86', 'from': 'n_i8', 'to': 'n_m5', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_87', 'from': 'n_m2', 'to': 'n_i3', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_88', 'from': 'n_i3', 'to': 'n_m2', 'length': None,
                  'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_89', 'from': 'n_s14', 'to': 'n_i7', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_90', 'from': 'n_i7', 'to': 'n_s14', 'length': None,
                  'numLanes': 1, 'type': 'edgeType'},
                 {'id': 'e_91', 'from': 'n_r5', 'to': 'n_r7', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(5.08 + 0.58 * cos(t), 1.17 + 0.58 * sin(t))
                            for t in linspace(1.08*pi, 1.60*pi, res)]
                  },
                 {'id': 'e_92', 'from': 'n_r3', 'to': 'n_r1', 'length': None,
                  'numLanes': 2, 'type': 'edgeType',
                  'shape': [(0.90 + 0.53 * cos(t), 0.93 + 0.53 * sin(t))
                            for t in linspace(3 * pi / 2 + 1.46 / 0.53,
                                              3.46 * pi / 2 + 1.46/0.53, res)]
                  },
                 {'id': 'e_93', 'from': 'n_s7_l', 'to': 'n_s7_r',
                  'length': None, 'numLanes': 2, 'type': 'edgeType'},
                 {'id': 'e_94', 'from': 'n_s7_r', 'to': 'n_s7_l',
                  'length': None, 'numLanes': 2, 'type': 'edgeType'}
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

            # fix junction overlapping issue
            junctions = {'e_8_b': 2,
                         'e_17': 18,
                         'e_8_u': 8,
                         'e_1': 9.9,
                         'e_29_u': 13,
                         'e_21': 5,
                         'e_91': 5,
                         'e_63': 26,
                         'e_65': 5,
                         'e_66': 30,
                         'e_32': 12,
                         'e_51': 8,
                         'e_37': 5,
                         'e_52': 5,
                         'e_18': 12,
                         'e_24': 12,
                         'e_13': 10,
                         'e_4': 10,
                         'e_36': 8,
                         'e_53': 49
                         }
            if edge['id'] in junctions:
                edge['length'] = str(junctions[edge['id']])

        return edges

    def specify_connections(self, net_params):
        """See parent class."""
        conn = []
        # connect lanes at bottlenecks
        num_lanes = 2
        edges_from_b = ['e_13', 'e_42', 'e_60']  # order matters
        edges_to_b = ['e_14', 'e_44', 'e_69']
        for e_from, e_to in zip(edges_from_b, edges_to_b):
            for i in range(num_lanes):
                conn += [{
                    'from': e_from,
                    'to': e_to,
                    'fromLane': str(i),
                    'toLane': str(int(np.floor(i / 2)))
                }]
        # connect lanes at roundabout (order matters)
        edges_from_r = ['e_66', 'e_66', 'e_7', 'e_7', 'e_9', 'e_9']
        edges_to_r = ['e_91', 'e_63', 'e_17', 'e_8_b', 'e_10', 'e_92']
        for r_from, r_to in zip(edges_from_r, edges_to_r):
            for i in range(num_lanes):
                conn += [{
                    'from': r_from,
                    'to': r_to,
                    'fromLane': str(i),
                    'toLane': str(i)
                }]

        # split one lane to two lanes from e_68 to e_66
        conn += [{
            'from': 'e_68',
            'to': 'e_66',
            'fromLane': '0',
            'toLane': '0'
        }]

        # remove u-turn connections at n_m4 and n_s7_l
        edges_from_u = ['e_37', 'e_51', 'e_94', 'e_94', 'e_36']
        edges_to_u = ['e_29_u', 'e_29_u', 'e_52', 'e_51', 'e_93']
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
        """See parent class."""
        types = [{'id': 'edgeType', 'speed': repr(30)}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {'e_1': ['e_1'],
               'e_2': ['e_2'],
               'e_3': ['e_3'],
               'e_4': ['e_4'],
               'e_5': ['e_5'],
               'e_6': ['e_6'],
               'e_7': ['e_7'],
               'e_8_b': ['e_8_b'],
               'e_8_u': ['e_8_u'],
               'e_9': ['e_9'],
               'e_10': ['e_10'],
               'e_11': ['e_11'],
               'e_12': ['e_12'],
               'e_13': ['e_13'],
               'e_14': ['e_14'],
               'e_15': ['e_15'],
               'e_16': ['e_16'],
               'e_17': ['e_17'],
               'e_18': ['e_18'],
               'e_19': ['e_19'],
               'e_20': ['e_20'],
               'e_21': ['e_21'],
               'e_22': ['e_22'],
               'e_23': ['e_23'],
               'e_24': ['e_24'],
               'e_25': ['e_25'],
               'e_26': ['e_26'],
               'e_27': ['e_27'],
               'e_28_b': ['e_28_b'],
               'e_29_u': ['e_29_u'],
               'e_30': ['e_30'],
               'e_31': ['e_31'],
               'e_32': ['e_32'],
               'e_33': ['e_33'],
               'e_34': ['e_34'],
               'e_35': ['e_35'],
               'e_36': ['e_36'],
               'e_37': ['e_37'],
               'e_38': ['e_38'],
               'e_39': ['e_39'],
               'e_40': ['e_40'],
               'e_41': ['e_41'],
               'e_42': ['e_42'],
               'e_43': ['e_43'],
               'e_44': ['e_44'],
               'e_45': ['e_45'],
               'e_46': ['e_46'],
               'e_47': ['e_47'],
               'e_48': ['e_48'],
               'e_49': ['e_49'],
               'e_50': ['e_50'],
               'e_51': ['e_51'],
               'e_52': ['e_52'],
               'e_53': ['e_53'],
               'e_54': ['e_54'],
               'e_55': ['e_55'],
               'e_56': ['e_56'],
               'e_57': ['e_57'],
               'e_58': ['e_58'],
               'e_59': ['e_59'],
               'e_60': ['e_60'],
               'e_61': ['e_61'],
               'e_62': ['e_62'],
               'e_63': ['e_63'],
               'e_64': ['e_64'],
               'e_65': ['e_65'],
               'e_66': ['e_66'],
               'e_67': ['e_67'],
               'e_68': ['e_68'],
               'e_69': ['e_69'],
               'e_70': ['e_70'],
               'e_71': ['e_71'],
               'e_72': ['e_72'],
               'e_73': ['e_73'],
               'e_74': ['e_74'],
               'e_75': ['e_75'],
               'e_76': ['e_76'],
               'e_77': ['e_77'],
               'e_78': ['e_78'],
               'e_79': ['e_79'],
               'e_80': ['e_80'],
               'e_81': ['e_81'],
               'e_82': ['e_82'],
               'e_83': ['e_83'],
               'e_84': ['e_84'],
               'e_85': ['e_85'],
               'e_86': ['e_86'],
               'e_87': ['e_87'],
               'e_88': ['e_88'],
               'e_89': ['e_89'],
               'e_90': ['e_90'],
               'e_91': ['e_91'],
               'e_92': ['e_92'],
               'e_93': ['e_93'],
               'e_94': ['e_94']}

        return rts
