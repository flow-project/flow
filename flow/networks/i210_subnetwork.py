"""Contains the I-210 sub-network class."""
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

ADDITIONAL_NET_PARAMS = {
    # whether to include vehicle on the on-ramp
    "on_ramp": False,
    # whether to include the downstream slow-down edge in the network
    "ghost_edge": False,
}

EDGES_DISTRIBUTION = [
    # Main highway
    "ghost0",
    "119257914",
    "119257908#0",
    "119257908#1-AddedOnRampEdge",
    "119257908#1",
    "119257908#1-AddedOffRampEdge",
    "119257908#2",
    "119257908#3",

    # On-ramp
    "27414345",
    "27414342#0",
    "27414342#1-AddedOnRampEdge",

    # Off-ramp
    "173381935",
]


class I210SubNetwork(Network):
    """A network used to simulate the I-210 sub-network.

    Requires from net_params:

    * **on_ramp** : whether to include vehicle on the on-ramp
    * **ghost_edge** : whether to include the downstream slow-down edge in the
      network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import I210SubNetwork
    >>>
    >>> network = I210SubNetwork(
    >>>     name='I-210_subnetwork',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize the I210 sub-network scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        # The length of each edge and junction is a fixed term that can be
        # found in the xml file.
        self.length_with_ghost_edge = [
            ("ghost0", 573.08),
            (":300944378_0", 0.30),
            ("119257914", 61.28),
            (":300944379_0", 0.31),
            ("119257908#0", 696.97),
            (":300944436_0", 2.87),
            ("119257908#1-AddedOnRampEdge", 97.20),
            (":119257908#1-AddedOnRampNode_0", 3.24),
            ("119257908#1", 239.68),
            (":119257908#1-AddedOffRampNode_0", 3.24),
            ("119257908#1-AddedOffRampEdge", 98.50),
            (":1686591010_1", 5.46),
            ("119257908#2", 576.61),
            (":1842086610_1", 4.53),
            ("119257908#3", 17.49),
        ]

        super(I210SubNetwork, self).__init__(
            name=name,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "119257914": [
                (["119257914",
                  "119257908#0",
                  "119257908#1-AddedOnRampEdge",
                  "119257908#1",
                  "119257908#1-AddedOffRampEdge",
                  "119257908#2",
                  "119257908#3"], 1.0),
            ]
        }

        if net_params.additional_params["ghost_edge"]:
            rts.update({
                "ghost0": [
                    (["ghost0",
                      "119257914",
                      "119257908#0",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1),
                ],
            })

        if net_params.additional_params["on_ramp"]:
            rts.update({
                # Main highway
                "119257908#0": [
                    (["119257908#0",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1.0),
                ],
                "119257908#1-AddedOnRampEdge": [
                    (["119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1.0),
                ],
                "119257908#1": [
                    (["119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1.0),
                ],
                "119257908#1-AddedOffRampEdge": [
                    (["119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1.0),
                ],
                "119257908#2": [
                    (["119257908#2",
                      "119257908#3"], 1),
                ],
                "119257908#3": [
                    (["119257908#3"], 1),
                ],

                # On-ramp
                "27414345": [
                    (["27414345",
                      "27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1 - 9 / 321),
                    (["27414345",
                      "27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "173381935"], 9 / 321),
                ],
                "27414342#0": [
                    (["27414342#0",
                      "27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 1 - 20 / 421),
                    (["27414342#0",
                      "27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "173381935"], 20 / 421),
                ],
                "27414342#1-AddedOnRampEdge": [
                    (["27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "119257908#2",
                      "119257908#3"], 0.5),
                    (["27414342#1-AddedOnRampEdge",
                      "27414342#1",
                      "119257908#1-AddedOnRampEdge",
                      "119257908#1",
                      "119257908#1-AddedOffRampEdge",
                      "173381935"], 0.5),
                ],

                # Off-ramp
                "173381935": [
                    (["173381935"], 1),
                ],
            })

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        if self.net_params.additional_params["ghost_edge"]:
            # Collect the names of all the edges.
            edge_names = [
                e[0] for e in self.length_with_ghost_edge
                if not e[0].startswith(":")
            ]

            edge_starts = []
            for edge in edge_names:
                # Find the position of the edge in the list of tuples.
                edge_pos = next(
                    i for i in range(len(self.length_with_ghost_edge))
                    if self.length_with_ghost_edge[i][0] == edge
                )

                # Sum of lengths until the edge is reached to compute the
                # starting position of the edge.
                edge_starts.append((
                    edge,
                    sum(e[1] for e in self.length_with_ghost_edge[:edge_pos])
                ))

        elif self.net_params.additional_params["on_ramp"]:
            # TODO: this will incorporated in the future, if needed.
            edge_starts = []

        else:
            # TODO: this will incorporated in the future, if needed.
            edge_starts = []

        return edge_starts

    def specify_internal_edge_starts(self):
        """See parent class."""
        if self.net_params.additional_params["ghost_edge"]:
            # Collect the names of all the junctions.
            edge_names = [
                e[0] for e in self.length_with_ghost_edge
                if e[0].startswith(":")
            ]

            edge_starts = []
            for edge in edge_names:
                # Find the position of the edge in the list of tuples.
                edge_pos = next(
                    i for i in range(len(self.length_with_ghost_edge))
                    if self.length_with_ghost_edge[i][0] == edge
                )

                # Sum of lengths until the edge is reached to compute the
                # starting position of the edge.
                edge_starts.append((
                    edge,
                    sum(e[1] for e in self.length_with_ghost_edge[:edge_pos])
                ))

        elif self.net_params.additional_params["on_ramp"]:
            # TODO: this will incorporated in the future, if needed.
            edge_starts = []

        else:
            # TODO: this will incorporated in the future, if needed.
            edge_starts = []

        return edge_starts
