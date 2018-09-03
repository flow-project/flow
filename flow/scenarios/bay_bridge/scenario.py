"""Contains the Bay Bridge scenario class."""

from flow.scenarios.netfile.scenario import NetFileScenario


class BayBridgeScenario(NetFileScenario):
    """A scenario used to simulate the bottleneck portion of the Bay Bridge."""

    def generate_starting_positions(self, **kwargs):
        """
        See parent class.

        Vehicles are only placed in the edges of the Bay Bridge moving from
        Oakland to San Francisco.
        """
        self.initial_config.edges_distribution = [
            '236348360#1',
            '157598960',
            '11415208',
            '236348361',
            '11198599',
            '11198595.0',
            '11198595.656.0',
            '340686911#3',
            '23874736',
            '119057701',
            '517934789',
            '236348364',
            '124952171',
            "gneE0",
            "11198599",
            "124952182.0",
            '236348360#0',
            '497579295',
            '340686911#2.0.0',
            '340686911#1',
            '394443191',
            '322962944',
            "32661309#1.0",
            "90077193#1.777",
            "90077193#1.0",
            "90077193#1.812",
            "gneE1",
            "32661316",
            "4757680",
            "124952179",
            "119058993",
            "28413679",
            "11197898",
            "123741311",
            "123741303",
            "90077193#0",
            "28413687#1",
            "11197889",
            "123741382#0",
            "123741382#1",
            "gneE3",
            "340686911#0.54.0",
            "340686911#0.54.54.0",
            "340686911#0.54.54.127.0",
            "340686911#2.35",
        ]

        return super().generate_starting_positions(**kwargs)
