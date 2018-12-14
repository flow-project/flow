import unittest

from tests.setup_scripts import setup_bottlenecks
from flow.core.experiment import SumoExperiment


class TestBottleneck(unittest.TestCase):
    def test_it_runs(self):
        self.env, self.scenario = setup_bottlenecks()
        self.exp = SumoExperiment(self.env)
        self.exp.run(5, 50)


if __name__ == '__main__':
    unittest.main()
