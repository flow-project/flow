"""Contains the Bay Bridge toll scenario class."""

from flow.scenarios.netfile.scenario import NetFileScenario


class BayBridgeTollScenario(NetFileScenario):
    """A scenario used to simulate the Bay Bridge toll."""

    def generate_starting_positions(self, **kwargs):
        """
        See parent class.

        Vehicles are only placed in the edges of the Bay Bridge moving from
        Oakland to San Francisco.
        """
        self.initial_config.edges_distribution = [
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

        return super().generate_starting_positions(**kwargs)
