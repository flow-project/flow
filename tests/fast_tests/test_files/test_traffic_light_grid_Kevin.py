import unittest
import os

from tests.setup_scripts import ring_road_exp_setup, traffic_light_grid_mxn_exp_setup
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.experiment import Experiment
from flow.controllers.routing_controllers import GridRouter
from flow.controllers.car_following_models import IDMController

os.environ["TEST_FLAG"] = "True"


class Node:
    def __init__(self):
        self.row_num = 2
        self.col_num = 3

    def node_mapping(self):
        """Map nodes to edges.
        Returns a list of pairs (node, connected edges) of all inner nodes
        and for each of them, the 4 edges that leave this node.
        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}

        for row in range(self.row_num):
            for col in range(self.col_num):
                node_id = "center{}".format(row * self.col_num + col)

                top_edge_id = "left{}_{}".format(row + 1, col)
                bot_edge_id = "right{}_{}".format(row, col)
                right_edge_id = "top{}_{}".format(row, col + 1)
                left_edge_id = "bot{}_{}".format(row, col)

                mapping[node_id] = [left_edge_id, bot_edge_id,
                                    right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])


    def node_mapping_outer(self):
        """Map outer nodes, specifically, the start nodes, to edges.
        Returns a list of pairs (node, connected edges) of all outer nodes
        and for each of them, the edge that starts from this node and the edge that ends at this node..
        The nodes are listed in alphabetical order, and within that, edges are
        listed in order: [bot, right, top, left].
        """
        mapping = {}
        for col in range(self.col_num):
            for node_pos in (["bot"]):
                node_in_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_in_id] = "right{}_{}".format(0, col)

                node_out_id = "{}_col_long{}".format(node_pos, col)
                mapping[node_out_id] = "left{}_{}".format(0, col)

        for col in range(self.col_num):
            for node_pos in (["top"]):
                node_in_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_in_id] = "left{}_{}".format(self.row_num, col)

                node_out_id = "{}_col_short{}".format(node_pos, col)
                mapping[node_out_id] = "right{}_{}".format(self.row_num, col)

        # for row in range(self.row_num):
        #     for col in range(self.col_num):
        #         node_id = "center{}".format(row * self.col_num + col)
        #
        #         top_edge_id = "left{}_{}".format(row + 1, col)
        #         bot_edge_id = "right{}_{}".format(row, col)
        #         right_edge_id = "top{}_{}".format(row, col + 1)
        #         left_edge_id = "bot{}_{}".format(row, col)
        #
        #         mapping[node_id] = [left_edge_id, bot_edge_id,
        #                             right_edge_id, top_edge_id]

        return sorted(mapping.items(), key=lambda x: x[0])

    def _edges(self):
        """Build out the edges of the network.
        Edges join nodes to each other.
        Consider the following network with n = 2 rows and m = 3 columns,
        where the nodes are marked by 'x':

        y
        ^
        |
    3   -         x     x     x
        |         |     |     |
    2   -    x----x-----x-----x----x
        |         |     |     |
    1   -    x----x-----x-(*)-x----x
        |         |     |     |
    0   -         x     x     x
        |
        ----|-----|-----|-----|----|----> x
             0    1     2     3    4


        There are n * (m + 1) = 8 horizontal edges and m * (n + 1) = 9
        vertical edges, all that multiplied by two because each edge
        consists of two roads going in opposite directions traffic-wise.
        Edge ids take the format of "({from_node})--({to_node})".

        For example, on edge (*) the id of the bottom road (traffic
        going from left to right) is "(2.1)--(3.1)" and the id of the top road
        (traffic going from right to left) is "(3.1)--(2.1)".
        Returns
        -------
        list <dict>
            List of inner edges
        """
        edges = []

        x_max = self.col_num + 1 # x_max = 4
        y_max = self.row_num + 1 # y_max = 3

        def new_edge(from_node, to_node, orientation):
            return [{
                "id": str(from_node) + "--" + str(to_node),
                # "type": orientation,
                # "priority": 78,
                # "from": str(from_node),
                # "to": str(to_node),
                # "length": 10
            }]

        # Build the horizontal of edges
        for y in range(1, y_max):
            for x in range(x_max):
                left_node = "({}.{})".format(x, y)
                right_node = "({}.{})".format(x + 1, y)
                edges += new_edge(left_node, right_node,
                                  "horizontal")
                edges += new_edge(right_node, left_node,
                                  "horizontal")
        # Build the vertical edges
        for x in range(1, x_max):
            for y in range(y_max):
                bottom_node = "({}.{})".format(x, y)
                top_node = "({}.{})".format(x, y + 1)
                edges += new_edge(bottom_node, top_node,
                                  "vertical")
                edges += new_edge(top_node, bottom_node,
                                  "vertical")

        return edges



if __name__ == '__main__':
    a = Node()
    print(a._edges() == [{'id': '(0.1)--(1.1)'}, {'id': '(1.1)--(0.1)'}, {'id': '(1.1)--(2.1)'}, {'id': '(2.1)--(1.1)'}, {'id': '(2.1)--(3.1)'}, {'id': '(3.1)--(2.1)'}, {'id': '(3.1)--(4.1)'}, {'id': '(4.1)--(3.1)'}, {'id': '(0.2)--(1.2)'}, {'id': '(1.2)--(0.2)'}, {'id': '(1.2)--(2.2)'}, {'id': '(2.2)--(1.2)'}, {'id': '(2.2)--(3.2)'}, {'id': '(3.2)--(2.2)'}, {'id': '(3.2)--(4.2)'}, {'id': '(4.2)--(3.2)'}, {'id': '(1.0)--(1.1)'}, {'id': '(1.1)--(1.0)'}, {'id': '(1.1)--(1.2)'}, {'id': '(1.2)--(1.1)'}, {'id': '(1.2)--(1.3)'}, {'id': '(1.3)--(1.2)'}, {'id': '(2.0)--(2.1)'}, {'id': '(2.1)--(2.0)'}, {'id': '(2.1)--(2.2)'}, {'id': '(2.2)--(2.1)'}, {'id': '(2.2)--(2.3)'}, {'id': '(2.3)--(2.2)'}, {'id': '(3.0)--(3.1)'}, {'id': '(3.1)--(3.0)'}, {'id': '(3.1)--(3.2)'}, {'id': '(3.2)--(3.1)'}, {'id': '(3.2)--(3.3)'}, {'id': '(3.3)--(3.2)'}]
)
