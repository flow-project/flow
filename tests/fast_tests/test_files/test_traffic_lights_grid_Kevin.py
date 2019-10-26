import unittest
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np
import networkx as nx
from itertools import islice


def specify_routes():
    """Returns a dict representing all possible routes of the network via the "Multiple routes per edge" format.

    :param net_params:

    Returns
    -------
    routes_dict <list <tuple>>

    The format of routes_dict is as follows:

            routes_dict = {"se0": [(["edge0", "edge1", "edge2", "edge3"], 1)],
                           "se1": [(["edge1", "edge2", "edge3", "edge0"], 1)],
                           "se2": [(["edge2", "edge3", "edge0", "edge1"], 1)],
                           "se3": [(["edge3", "edge0", "edge1", "edge2"], 1)]}

            routes_dict = {"start_edge":
                                        [(Route A beginning with start_edge, Pr(route A)),
                                         (Route B beginning with start_edge, Pr(route B)]
                            ...
                            }

    """

    routes_dict = {}

    def generate_node_routes_list(k=5):
        """Create a list of k shortest routes for all possible pairs of source and destination outer nodes. By default,
        for any two pairs of source and destination nodes, generate k shortest routes.

        Returns
        -------
        routes_dict_nodes, a dict of dicts.

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

        (Here, sn is shorthand for "start node", and en is shorthand for "end node")

        routes_nodes_dict = {start_node:
                                {end_node: [rt1, rt2, rt3]}
                                                            ... }

        routes are written in terms of a sequence of nodes e.g. ['(0.1)', '(0.2)', '(1.2)']


        routes_dict = {"start_edge":
                            [(Route A beginning with start_edge, Pr(route A)),
                             (Route B beginning with start_edge, Pr(route B)]
                        ...
                    }

        # I'm going to use a non-hard coded method to convert the routes nodes dict to routes dict with edges i.e.
        # I'll need to connect up the start node with the next node in the sequence of nodes

        """

        routes_nodes_dict = {}
        g = nx.DiGraph()
        src_dst_nodes = []  # list of all outer nodes that function as starting (and ending) nodes

        x_max = 2 + 1
        y_max = 3 + 1

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
        for src in src_dst_nodes:
            routes_nodes_dict[src] = {}
            for dst in src_dst_nodes:
                if src != dst:  # In reality, one would probably want to be able to return to a particular node
                    curr_src_dict = routes_nodes_dict[src].get(dst, {})
                    curr_src_dict_lst = curr_src_dict.get(dst, [])
                    curr_src_dict_lst.extend(k_shortest_paths(g, src, dst, k))
                    routes_nodes_dict[src][dst] = curr_src_dict_lst
                    # if src != dst: do we want cars to be able to return to the same edge it came from?
        return routes_nodes_dict

    def node_route_to_edge_route(node_route):
        """Convert a shortest path specified by a sequence of nodes to a shortest path specified by a sequence of edges
        Returns a list of edges"""

        edge_route = []
        for node_index in range(len(node_route) - 1):
            curr_node = node_route[node_index]
            next_node = node_route[node_index + 1]
            curr_edge = curr_node + "--" + next_node
            edge_route.append(curr_edge)

        return edge_route

    node_routes = generate_node_routes_list(5)  # for all src dst node pairs, generate the top 5 shortest paths
    # TODO: convert node pairings into edge pairings - take into account that some routes have a fixed number of routes

    """(Here, sn is shorthand for "start node", and en is shorthand
    for "end node")

    routes_nodes_dict = {start_node:
                             {end_node: [rt1, rt2, rt3]}
                                 ...}

    e.g.rt1 = ['(0.1)', '(0.2)', '(1.2)']

    routes_dict = {"start_edge":
                             [(Route A beginning with start_edge, Pr(route A)),
                              (Route B beginning with start_edge, Pr(route B)]
    ...
    }

    # I'm going to use a non-hard coded method to convert the routes nodes dict to routes dict with edges i.e. 
    # I'll need to connect up the start node with the next node in the sequence of nodes"""

    for start_node in node_routes:
        start_node_routes_dict = node_routes[start_node]
        for end_node in start_node_routes_dict:
            start_end_node_routes_list = node_routes[start_node][end_node]
            num_rts = len(
                start_end_node_routes_list)  # this is a hard coded method of calculating probabilites, TODO: change to non-hard coded method
            for node_route in start_end_node_routes_list:
                #  1. Convert node specified route to an edge specified route
                #  2. Calculate required probabilities
                edge_route = node_route_to_edge_route(node_route)
                start_edge = edge_route[0]
                curr_start_edge_routes = routes_dict.get(start_edge, [])
                curr_start_edge_routes.append((edge_route, 1 / num_rts))
                routes_dict[start_edge] = curr_start_edge_routes

    return routes_dict

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
    for src in src_dst_nodes:
            routes_dict[src] = {}
            for dst in src_dst_nodes:
                if src != dst:  # In reality, one would probably want to be able to return to a particular node
                    curr_src_dict = routes_dict[src].get(dst, {})
                    curr_src_dict_lst = curr_src_dict.get(dst, [])
                    curr_src_dict_lst.extend(k_shortest_paths(g, src, dst, k))
                    routes_dict[src][dst] = curr_src_dict_lst
                    # if src != dst: do we want cars to be able to return to the same edge it came from?

    return routes_dict

class MyTestCase(unittest.TestCase):
    # def test_generate_node_routes_list(self):
    #     node_routes_dict = generate_node_routes_list(5)
    #     print(node_routes_dict)
    #     src = "({}.{})".format(0, 1)
    #     dst = "({}.{})".format(4, 2)
    #     print(node_routes_dict[src][dst])

    def test_specify_routes(self):
        routes_dict = specify_routes()

        print(routes_dict)



if __name__ == '__main__':
    unittest.main()
