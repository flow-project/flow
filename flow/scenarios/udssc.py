import xml.etree.ElementTree as ET

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights

import numpy as np
from numpy import pi, sin, cos, linspace, sqrt


ADDITIONAL_NET_PARAMS = {
    # radius of the loops
    "ring_radius": 50,
    # length of the straight edges connected the outer loop to the inner loop
    "lane_length": 75,
    # length of the merge next to the roundabout
    "merge_length": 15,
    # number of lanes in the inner loop. DEPRECATED. DO NOT USE
    "inner_lanes": 3,
    # number of lanes in the outer loop. DEPRECATED. DO NOT USE
    "outer_lanes": 2, 
    # max speed limit in the roundabout
    "roundabout_speed_limit": 8,
    # max speed limit in the roundabout
    "outside_speed_limit": 15,
    # resolution of the curved portions
    "resolution": 40,
    # num lanes
    "lane_num": 1,
}


class UDSSCMergingScenario(Scenario):
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initializes a two loop scenario where one loop merging in and out of
        the other.

        Requires from net_params:
        - ring_radius: radius of the loops
        - lane_length: length of the straight edges connected the outer loop to
          the inner loop
        - inner_lanes: number of lanes in the inner loop
        - outer_lanes: number of lanes in the outer loop
        - speed_limit: max speed limit in the network
        - resolution: resolution of the curved portions

        See Scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
        # <-- 
        self.es = {}

        # -->
        radius = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]

        self.junction_length = 0.3
        self.intersection_length = 25.5  # calibrate when the radius changes

        net_params.additional_params["length"] = \
            2 * x + 2 * pi * radius + \
            2 * self.intersection_length + 2 * self.junction_length

        num_vehicles = vehicles.num_vehicles
        num_merge_vehicles = sum("merge" in vehicles.get_state(veh_id, "type")
                                 for veh_id in vehicles.get_ids())
        self.n_inner_vehicles = num_merge_vehicles
        self.n_outer_vehicles = num_vehicles - num_merge_vehicles

        radius = net_params.additional_params["ring_radius"]
        length_loop = 2 * pi * radius
        self.length_loop = length_loop
        self.roundabout_speed_limit = net_params.additional_params["roundabout_speed_limit"]
        self.outside_speed_limit = net_params.additional_params["outside_speed_limit"]
        self.lane_num = net_params.additional_params["lane_num"]

        super().__init__(name, vehicles, net_params,
                         initial_config, traffic_lights)

    

    def specify_edge_starts(self):
        """
        See parent class
        """
        edge_dict = {}
        absolute = 0
        prev_edge = 0
        for edge in self.specify_absolute_order():
            
            if edge.startswith(":"):
                # absolute += float(self.edge_info[edge]["length"])
                absolute += float(self.edge_length(edge))
                continue
            new_x = absolute + prev_edge #+ prev_internal
            edge_dict[edge] = new_x
            # prev_edge = float(self.edge_info[edge]["length"])
            prev_edge = float(self.edge_length(edge))
            absolute = new_x
        self.es.update(edge_dict)

        edgestarts = [ #len of prev edge + total prev (including internal edge len)
            ("right", edge_dict["right"]),
            ("top", edge_dict["top"]), 
            ("left", edge_dict["left"]),
            ("bottom", edge_dict["bottom"]),
            ("inflow_1", edge_dict["inflow_1"]),
            ("merge_in_1", edge_dict["merge_in_1"]),
            ("merge_out_0", edge_dict["merge_out_0"]),
            ("outflow_0", edge_dict["outflow_0"]),
            ("inflow_0", edge_dict["inflow_0"]),
            ("merge_in_0", edge_dict["merge_in_0"]),
            ("merge_out_1", edge_dict["merge_out_1"]),
            ("outflow_1", edge_dict["outflow_1"]),
        ]

        # import ipdb; ipdb.set_trace()
        return edgestarts

    def specify_internal_edge_starts(self):
        """
        See parent class
        """
        edge_dict = {}
        absolute = 0
        prev_edge = 0
        for edge in self.specify_absolute_order(): # each edge = absolute + len(prev edge) + len(prev internal edge)
            
            if not edge.startswith(":"):
                # absolute += float(self.edge_info[edge]["length"])
                absolute += float(self.edge_length(edge))
                continue
            new_x = absolute + prev_edge
            edge_dict[edge] = new_x
            # prev_edge = float(self.edge_info[edge]["length"])
            prev_edge = float(self.edge_length(edge))
            absolute = new_x

        if self.lane_num == 2: 
        # two lane 
            internal_edgestarts = [ # in increasing order
                (":a_2", edge_dict[":a_2"]),
                (":b_2", edge_dict[":b_2"]),
                (":c_2", edge_dict[":c_2"]),
                (":d_2", edge_dict[":d_2"]),
                (":g_3", edge_dict[":g_3"]),
                (":b_0", edge_dict[":b_0"]),
                (":e_2", edge_dict[":e_2"]),
                (":e_0", edge_dict[":e_0"]),
                (":d_0", edge_dict[":d_0"]),
                (":g_0", edge_dict[":g_0"]),
            ]
        elif self.lane_num == 1:
        # one lane
            internal_edgestarts = [ # in increasing order
                (":a_1", edge_dict[":a_1"]),
                (":b_1", edge_dict[":b_1"]),
                (":c_1", edge_dict[":c_1"]),
                (":d_1", edge_dict[":d_1"]),
                (":g_2", edge_dict[":g_2"]),
                (":a_0", edge_dict[":a_0"]),
                (":b_0", edge_dict[":b_0"]),
                (":e_1", edge_dict[":e_1"]),
                (":e_0", edge_dict[":e_0"]),
                (":c_0", edge_dict[":c_0"]),
                (":d_0", edge_dict[":d_0"]),
                (":g_0", edge_dict[":g_0"]),
            ]
        self.es.update(edge_dict)

        return internal_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """
        See parent class

        Vehicles with the prefix "merge" are placed in the merge ring,
        while all other vehicles are placed in the ring.
        """
        x0 = initial_config.x0
        # changes to x0 in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "x0" in kwargs:
            x0 = kwargs["x0"]

        random_scale = \
            self.initial_config.additional_params.get("gaussian_scale", 0)

        bunching = initial_config.bunching
        # changes to bunching in kwargs suggests a switch in between rollouts,
        #  and so overwrites anything in initial_config
        if "bunching" in kwargs:
            bunching = kwargs["bunching"]

        merge_bunching = 0
        if "merge_bunching" in initial_config.additional_params:
            merge_bunching = initial_config.additional_params["merge_bunching"]

        num_vehicles = self.vehicles.num_vehicles
        num_merge_vehicles = \
            sum("merge" in self.vehicles.get_state(veh_id, "type")
                for veh_id in self.vehicles.get_ids())

        radius = self.net_params.additional_params["ring_radius"]
        lane_length = self.net_params.additional_params["lane_length"]

        startpositions = []
        startlanes = []
        length_loop = 2 * pi * radius

        try:
            increment_loop = \
                (self.length_loop - bunching) \
                * self.net_params.additional_params["inner_lanes"] \
                / (num_vehicles - num_merge_vehicles)

            # x = [x0] * initial_config.lanes_distribution
            if self.initial_config.additional_params.get("ring_from_right",
                                                         False):
                x = [dict(self.edgestarts)["right"]] * \
                    self.net_params.additional_params["inner_lanes"]
            else:
                x = [x0] * self.net_params.additional_params["inner_lanes"]
            car_count = 0
            lane_count = 0
            while car_count < num_vehicles - num_merge_vehicles:
                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                # ensures that vehicles are not placed in an internal junction
                while pos[0] in dict(self.internal_edgestarts).keys():
                    # find the location of the internal edge in
                    # total_edgestarts, which has the edges ordered by position
                    edges = [tup[0] for tup in self.total_edgestarts]
                    indx_edge = next(i for i, edge in enumerate(edges)
                                     if edge == pos[0])

                    # take the next edge in the list, and place the car at the
                    # beginning of this edge
                    if indx_edge == len(edges)-1:
                        next_edge_pos = self.total_edgestarts[0]
                    else:
                        next_edge_pos = self.total_edgestarts[indx_edge+1]

                    x[lane_count] = next_edge_pos[1]
                    pos = (next_edge_pos[0], 0)

                startpositions.append(pos)
                startlanes.append(lane_count)

                x[lane_count] = \
                    (x[lane_count] + increment_loop
                     + random_scale * np.random.randn()) % length_loop

                # increment the car_count and lane_num
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                if lane_count >= \
                        self.net_params.additional_params["inner_lanes"]:
                    lane_count = 0
        except ZeroDivisionError:
            pass

        length_merge = pi * radius + 2 * lane_length
        try:
            increment_merge = \
                (length_merge - merge_bunching) * \
                initial_config.lanes_distribution / num_merge_vehicles

            if self.initial_config.additional_params.get("merge_from_top",
                                                         False):
                x = [dict(self.edgestarts)["top"] - x0] * \
                    self.net_params.additional_params["outer_lanes"]
            else:
                x = [dict(self.edgestarts)["bottom"] - x0] * \
                    self.net_params.additional_params["outer_lanes"]
            car_count = 0
            lane_count = 0
            while car_count < num_merge_vehicles:
                # collect the position and lane number of each new vehicle
                pos = self.get_edge(x[lane_count])

                # ensures that vehicles are not placed in an internal junction
                while pos[0] in dict(self.internal_edgestarts).keys():
                    # find the location of the internal edge in
                    # total_edgestarts, which has the edges ordered by position
                    edges = [tup[0] for tup in self.total_edgestarts]
                    indx_edge = next(i for i, edge in enumerate(edges)
                                     if edge == pos[0])

                    # take the next edge in the list, and place the car at the
                    # beginning of this edge
                    if indx_edge == len(edges)-1:
                        next_edge_pos = self.total_edgestarts[0]
                    else:
                        next_edge_pos = self.total_edgestarts[indx_edge+1]

                    x[lane_count] = next_edge_pos[1]
                    pos = (next_edge_pos[0], 0)

                startpositions.append(pos)
                startlanes.append(lane_count)

                if self.initial_config.additional_params.get(
                        "merge_from_top", False):
                    x[lane_count] = x[lane_count] - increment_merge + \
                        random_scale*np.random.randn()
                else:
                    x[lane_count] = x[lane_count] + increment_merge + \
                        random_scale*np.random.randn()

                # increment the car_count and lane_num
                car_count += 1
                lane_count += 1
                # if the lane num exceeds the number of lanes the vehicles
                # should be distributed on in the network, reset
                # if lane_count >= self.initial_config.lane_distribution
                if lane_count >= \
                        self.net_params.additional_params["outer_lanes"]:
                    lane_count = 0

        # CHANGES START
        # startpositions = [('right', 2.87), ('inflow_1', 10.0)]
        # CHANGES END
        except ZeroDivisionError:
            pass
        # First one corresponds to the IDM,
        # second one corresponds to the RL 
        if 'rl_0' in self.vehicles.get_ids(): # HARDCODE ALERT
            # startpositions = [('inflow_0', 10), ('right', 10)]
            startpositions = [('outflow_0', 0), ('outflow_0', 10)]
        else: 
            startpositions = [('inflow_0', 10)]
        
        return startpositions, startlanes

    def specify_absolute_order(self):
        if self.lane_num == 2: 
        
            return [":a_2", "right", ":b_2", "top", ":c_2",
                    "left", ":d_2", "bottom", "inflow_1",
                    ":g_3", "merge_in_1", ":a_0", ":b_0",
                    "merge_out_0", ":e_2", "outflow_0", "inflow_0",
                    ":e_0", "merge_in_0", ":c_0", ":d_0",
                    "merge_out_1", ":g_0", "outflow_1" ]
        elif self.lane_num == 1: 
        # one lane
            return [":a_1", "right", ":b_1", "top", ":c_1",
                    "left", ":d_1", "bottom", "inflow_1",
                    ":g_2", "merge_in_1", ":a_0", ":b_0",
                    "merge_out_0", ":e_1", "outflow_0", "inflow_0",
                    ":e_0", "merge_in_0", ":c_0", ":d_0",
                    "merge_out_1", ":g_0", "outflow_1" ]


    def specify_nodes(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        m = self.net_params.additional_params["merge_length"]

        roundabout_type = "priority"
        default = "priority"

        nodes = [{"id": "a",   "x": repr(0),  "y": repr(-r), "type": roundabout_type},
                 {"id": "b",   "x": repr(0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "c",   "x": repr(-0.5 * r),  "y": repr(sqrt(3)/2 * r), "type": roundabout_type},
                 {"id": "d",   "x": repr(-r), "y": repr(0), "type": roundabout_type},
                 {"id": "e",   "x": repr(0), "y": repr(r + m), "type": default},
                 {"id": "f",   "x": repr(0), "y": repr(r + m + x), "type": default},
                 {"id": "g",   "x": repr(-r - m), "y": repr(-r - 0.1*r), "type": default},
                 {"id": "h",   "x": repr(-r - m - x), "y": repr(-r - 0.2*r), "type": default},
                ]

        return nodes

    def specify_edges(self, net_params):
        """
        See parent class
        """
        r = net_params.additional_params["ring_radius"]
        x = net_params.additional_params["lane_length"]
        circumference = 2 * pi * r
        lanes = repr(net_params.additional_params["lane_num"])
        
        resolution = net_params.additional_params["resolution"]

        length = net_params.additional_params["length"]
        # edgelen = length / 4.
        circ = 2 * pi * r
        twelfth = circ / 12
        edges = [
            {"id": "bottom",
             "type": "edgeType_hi",
             "from": "d",
             "to": "a",
             "numLanes": lanes,
             "length": repr(twelfth * 3),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi, -pi/2 , resolution)])},

            {"id": "right",
             "type": "edgeType_hi",
             "from": "a",
             "to": "b",
             "numLanes": lanes,
             "length": repr(twelfth * 5),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(-pi / 2,pi/3, resolution)])},

            {"id": "top",
             "type": "edgeType_hi",
             "from": "b",
             "to": "c",
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(pi/3, 2*pi/3, resolution)])},

            {"id": "left",
             "type": "edgeType_hi",
             "from": "c",
             "to": "d", 
             "numLanes": lanes,
             "length": repr(twelfth * 2),
             "shape": " ".join(["%.2f,%.2f" % (r * cos(t), r * sin(t))
                                for t in linspace(2*pi/3, pi, resolution)])},

            {"id": "merge_out_0",
             "type": "edgeType_lo",
             "from": "b",
             "to": "e",
             "numLanes": lanes,
            },

            {"id": "merge_in_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "c",
             "numLanes": lanes,
            },

            {"id": "outflow_0",
             "type": "edgeType_lo",
             "from": "e",
             "to": "f",
             "numLanes": lanes,
            },

            {"id": "inflow_0",
             "type": "edgeType_lo",
             "from": "f",
             "to": "e",
             "numLanes": lanes,
            },

            {"id": "merge_out_1",
             "type": "edgeType_lo",
             "from": "d",
             "to": "g",
             "numLanes": lanes,
            },
            
            {"id": "merge_in_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "a",
             "numLanes": lanes,
            },

            {"id": "outflow_1",
             "type": "edgeType_lo",
             "from": "g",
             "to": "h",
             "numLanes": lanes,
            },

            {"id": "inflow_1",
             "type": "edgeType_lo",
             "from": "h",
             "to": "g",
             "numLanes": lanes,
            },
        ]

        return edges

    def specify_types(self, net_params):
        """
        See parent class
        """
        types = [{"id": "edgeType_hi",
                  "speed": repr(net_params.additional_params.get("roundabout_speed_limit")),
                  "priority": repr(2)},
                 {"id": "edgeType_lo",
                  "speed": repr(net_params.additional_params.get("outside_speed_limit")),
                  "priority": repr(1)}]
        return types

    def specify_routes(self, net_params):
        """
        See parent class
        """

        rts = {"top": {"top": ["top", "left", "bottom", "right"]},
               "left": {"left": ["left", "bottom", "right", "top"]},
               "bottom": {"bottom": ["bottom", "right", "top", "left"]},
               "right": {"right": ["right", "top", "left", "bottom"]},

               "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"]}, # added
               "inflow_0": {"inflow_1_1": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"]},

            #    "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
            #    "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

               "outflow_1": {"outflow_1": ["outflow_1"]},
               "outflow_0": {"outflow_0": ["outflow_0"]}
               }
               
        rts = {"top": ["top", "left", "bottom", "right"],
               "left": ["left", "bottom", "right", "top"],
               "bottom": ["bottom", "right", "top", "left"],
               "right": ["right", "top", "left", "bottom"],
               "inflow_1": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"], # added
               "inflow_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],

            #    "inflow_1": {"inflow_1_0": ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"]}, # added
            #    "inflow_0": {"inflow_0_0": ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"],
            #                 "inflow_1_1": ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"]},

               "outflow_1": ["outflow_1"],
               "outflow_0": ["outflow_0"]
               }
        # routes = Routes()
        # routes.add("top_0", ["top", "left", "bottom", "right"])
        # routes.add("left_0", ["left", "bottom", "right", "top"])
        # routes.add("bottom_0", ["bottom", "right", "top", "left"])
        # routes.add("right_0", ["right", "top", "left", "bottom"])
        # routes.add("inflow_1_0", ["inflow_1", "merge_in_1", "right", "top", "left", "merge_out_1", "outflow_1"])
        # # routes.add("inflow_1_1", ["inflow_1", "merge_in_1", "right", "merge_out_0", "outflow_0"])
        # routes.add("inflow_0_0", ["inflow_0", "merge_in_0", "left", "merge_out_1", "outflow_1"])
        # # routes.add("inflow_0_1", ["inflow_0", "merge_in_0", "left", "bottom", "right", "merge_out_0", "outflow_0"])
        # routes.add("outflow_1", ["outflow_1"])
        # routes.add("outflow_0", ["outflow_0"])
        # return routes

        return rts