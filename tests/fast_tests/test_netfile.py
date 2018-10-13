import unittest
import os
from tests.setup_scripts import setup_bottlenecks
from flow.core.experiment import SumoExperiment
from flow.scenarios.netfile.gen import NetFileGenerator


class TestNetfile(unittest.TestCase):
    def test_it_runs(self):
        self.env, self.scenario = setup_bottlenecks()
        l=os.path.join(self.scenario.generator.cfg_path, self.scenario.generator.roufn)
        gen=NetFileGenerator(None,None)
        print(l)
        routes=gen._import_routes_from_net(l)
        print(routes)
        raise AssertionError
        self.exp = SumoExperiment(self.env, self.scenario)
        self.exp.run(5, 50)


if __name__ == '__main__':
    unittest.main()
