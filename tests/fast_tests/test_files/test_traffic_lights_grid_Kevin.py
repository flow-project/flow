import unittest
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np
import networkx as nx
from itertools import islice

def generate_node_routes_list(k=5):
    """Create a list of k shortest routes for all possible pairs of source and destination outer nodes. By default,
    for any two pairs of source and destination nodes, generate k shortest routes.

    Returns
    -------
    routes_dict, a dict of dicts.

    The keys to the outer dict are the names of the starting nodes e.g "(0.1)". The value of the outer dict are
    more dicts. e.g. routes_dict["(0.1)"] returns another dict. Let's call this dict "(0.1)-routes". The keys to
    this "(0.1)-routes" dict are the names of an ending node e.g. "(0.4)". The value for a "(0.1)-routes" dict is
    a list of all routes (in terms of a sequence of nodes in a list). This list is ordered from the path with the
    shortest number of nodes to the kth shortest path.

    routes_dict should have this form:

    routes_dict = {"sn0": {"en0": [route_1(sn0 - en0), route_2(sn0 - en0), route_3(sn0 - en0), ... route_k(sn0, en0)],
                           "en1": [route_1(sn0 - en1), route_2(sn0 - en1), route_3(sn0 - en1), ... route_k(sn0, en1)],
                           "en2": [route_1(sn0 - en2), route_2(sn0 - en2), route_3(sn0 - en2), ... route_k(sn0, en2)],
                                ...
                           "enj": [route_1(sn0 - en2), route_2(sn0 - en2), route_3(sn0 - en2), ... route_k(sn0, en2)]},

                   "sn1": {"en0": [route_1(sn1 - en0), route_2(sn1 - en0), route_3(sn1 - en0), ... route_k(sn1, en0)],
                           "en1": [route_1(sn1 - en1), route_2(sn1 - en1), route_3(sn1 - en1), ... route_k(sn1, en1)],
                           "en2": [route_1(sn1 - en2), route_2(sn1 - en2), route_3(sn1 - en2), ... route_k(sn1, en2)],
                                ...
                           "enj": [route_1(sn1 - enj), route_2(sn1 - enj), route_3(sn1 - enj), ... route_k(sn1, enj)]},
                    ...
                   }
    """

    routes_dict = {}
    g = nx.DiGraph()
    src_dst_nodes = []  # list of all outer nodes that function as starting (and ending) nodes
    col_num = 3
    row_num = 2
    x_max = col_num + 1
    y_max = row_num + 1

    def k_shortest_paths(G, source, target, len_k, weight=None):
        """Takes in a graph, source and target and returns a list of k lists. Each inner list contains a sequence
         of nodes that form a path between a source and target node.

        Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.
        simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths"""

        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), len_k))

    # Generate all nodes in the grid for the graph
    for x in range(x_max + 1):
        for y in range(y_max + 1):
            if (x, y) not in [(0, 0), (x_max, 0), (0, y_max), (x_max, y_max)]:
                g.add_node("({}.{})".format(x, y))
                if x == x_max or x == 0 or y == y_max or y == 0:
                    src_dst_nodes += ["({}.{})".format(x, y)]

    # Build all the 'horizontal' edges for the graph
    for y in range(1, y_max):
        for x in range(x_max):
            left_node = "({}.{})".format(x, y)
            right_node = "({}.{})".format(x + 1, y)
            g.add_edge(left_node, right_node, weight=1)
            g.add_edge(right_node, left_node, weight=1)

    # Build all the 'vertical' edges for the graph
    for x in range(1, x_max):
        for y in range(y_max):
            bottom_node = "({}.{})".format(x, y)
            top_node = "({}.{})".format(x, y + 1)
            g.add_edge(bottom_node, top_node, weight=1)
            g.add_edge(top_node, bottom_node, weight=1)

    # Loop through all possible source and destination nodes to generate list of node sequences that represent possible paths
    for src in src_dst_nodes:  # e.g [n1, n2, n3, n4, n5]
        routes_dict[src] = {}
        for dst in src_dst_nodes:
            curr_src_dict = routes_dict[src].get(dst, {})
            curr_src_dict_lst = curr_src_dict.get(dst, [])
            curr_src_dict_lst.extend(k_shortest_paths(g, src, dst, k))
            routes_dict[src][dst] = curr_src_dict_lst
            # if src != dst: do we want cars to be able to return to the same edge it came from?

    return routes_dict

class MyTestCase(unittest.TestCase):
    def test_generate_node_routes_list(self):
        node_routes_dict = generate_node_routes_list(5)
        src = "({}.{})".format(0, 1)
        dst = "({}.{})".format(4, 2)
        print(node_routes_dict[src][dst])


if __name__ == '__main__':
    unittest.main()
