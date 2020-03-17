import unittest
import sumolib

from flow.core.kernel.kernel import Kernel
from flow.core.params import SumoParams, VehicleParams, NetParams

from flow.networks.ring import RingNetwork


class TestTraciSimulation(unittest.TestCase):
    """Explicit Tests for the traci kernel."""

    def setUp(self):
        self.network = RingNetwork(
            name='ring_road_test',
            vehicles=VehicleParams(),
            net_params=NetParams(
                additional_params={
                    'length': 230,
                    'lanes': 1,
                    'speed_limit': 30,
                    'resolution': 40
                },
            )
        )

    def test_setup_teardown_traci_simulation(self):
        """Tests The setup and teardown of the traci kernel."""

        self.dummy_sim_params = SumoParams()
        self.dummy_sim_params.port = sumolib.miscutils.getFreeSocketPort()
        self.k = Kernel('traci', self.dummy_sim_params)
        self.k.network.generate_network(self.network)
        # initialize the simulation using the simulation kernel. This will use
        # the network kernel as an input in order to determine what network
        # needs to be simulated.
        self.k.simulation.start_simulation(network=self.k.network, sim_params=self.dummy_sim_params)

        assert self.k.simulation.sumo_proc is not None, "Sumo proc was not created!"

        self.k.simulation.teardown_sumo()

    def test_setup_teardown_traci_simulation_libsumo(self):
        """Tests The setup and teardown of the traci kernel."""

        self.dummy_sim_params = SumoParams(use_libsumo=True)
        self.dummy_sim_params.port = sumolib.miscutils.getFreeSocketPort()
        self.k = Kernel('traci', self.dummy_sim_params)
        self.k.network.generate_network(self.network)

        # initialize the simulation using the simulation kernel. This will use
        # the network kernel as an input in order to determine what network
        # needs to be simulated.
        self.k.simulation.start_simulation(network=self.k.network, sim_params=self.dummy_sim_params)

        assert self.k.simulation.sumo_proc is not None, "Sumo proc was not created!"

        self.k.simulation.teardown_sumo()
