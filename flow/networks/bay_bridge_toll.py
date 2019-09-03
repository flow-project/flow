"""Contains the Bay Bridge toll network class."""

from flow.networks.base import Network

# Use this to ensure that vehicles are only placed in the edges of the Bay
# Bridge moving from Oakland to San Francisco.
EDGES_DISTRIBUTION = [
    '157598960',
    '11198599',
    '11198595.0',
    '11198595.656.0',
    '124952171',
    "gneE0",
    "11198599",
    "124952182.0",
    '340686911#2.0.0',
    '340686911#1',
    "32661309#1.0",
    "90077193#1.777",
    "90077193#1.0",
    "90077193#1.812",
    "gneE1",
    "124952179",
    "gneE3",
    "340686911#0.54.0",
    "340686911#0.54.54.0",
    "340686911#0.54.54.127.0",
    "340686911#2.35",
]


class BayBridgeTollNetwork(Network):
    """A network used to simulate the bottleneck portion of the Bay Bridge.

    The bay bridge was originally imported from OpenStreetMap and subsequently
    modified to more closely match the network geometry of the actual Bay
    Bridge. As opposed to BayBridgeNetwork, this network places vehicles on a
    reduced portion of the Bay Bridge in order to reduce the number of vehicles
    that need to be simulated.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import BayBridgeTollNetwork
    >>>
    >>> network = BayBridgeTollNetwork(
    >>>     name='bay_bridge_toll',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def specify_routes(self, net_params):
        """See parent class.

        Routes for vehicles moving through the bay bridge from Oakland to San
        Francisco.
        """
        rts = {
            "11198593": ["11198593", "11198595.0"],
            "157598960": ["157598960", "11198595.0"],
            "11198595.0": ["11198595.0", "11198595.656.0"],
            "11198595.656.0": ["11198595.656.0", "gneE5"],
            "gneE5": ["gneE5", "340686911#2.0.13"],
            "124952171": ["124952171", "11198599"],
            "340686911#1": ["340686911#1", "340686911#2.0.0"],
            "340686911#2.0.0": ["340686911#2.0.0", "340686911#2.0.13"],
            "340686911#0.54.54.127.74":
            ["340686911#0.54.54.127.74", "340686911#1"],
            "340686911#2.0.13": ["340686911#2.0.13", "340686911#2.35"],
            "340686911#2.35": ["340686911#2.35"],
            "393649534": ["393649534", "124952179"],
            "32661316": ["32661316", "124952179"],
            "124952179": ["124952179", "157598960"],
            "124952179_1": ["124952179", "124952171"],
            "4757680": ["4757680", "32661309#0"],
            "90077193#0": ["90077193#0", "90077193#1.0"],
            "11198599": ["11198599", "124952182.0"],
            "124952182.0": ["124952182.0", "gneE0"],
            "gneE0": ["gneE0", "90077193#1.777"],
            "90077193#1.777": ["90077193#1.777", "90077193#1.812"],
            "32661309#0": ["32661309#0", "32661309#1.0"],
            "32661309#1.0": ["32661309#1.0", "gneE1"],
            "gneE1": ["gneE1", "90077193#1.812"],
            "90077193#1.0": ["90077193#1.0", "90077193#1.777"],
            "90077193#1.812": ["90077193#1.812", "gneE3"],
            "gneE3": ["gneE3", "340686911#0.54.0"],
            "340686911#0.54.0": ["340686911#0.54.0", "340686911#0.54.54.0"],
            "340686911#0.54.54.0":
            ["340686911#0.54.54.0", "340686911#0.54.54.127.0"],
            "340686911#0.54.54.127.0":
            ["340686911#0.54.54.127.0", "340686911#0.54.54.127.74"],
        }

        return rts
