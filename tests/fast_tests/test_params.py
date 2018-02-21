import unittest
from flow.core.params import SumoLaneChangeParams
import os
os.environ["TEST_FLAG"] = "True"


class TestSumoLaneChangeParams(unittest.TestCase):
    """
    Tests that SumoLaneChangeParams only returns parameters that are valid to
    the given LC model.
    """

    def runTest(self):
        # test for LC2013
        lc_params_1 = SumoLaneChangeParams(model="LC2013")
        attributes_1 = list(lc_params_1.controller_params.keys())
        # TODO: modify with all elements once the fix is added to sumo
        expected_attributes_1 = ["laneChangeModel", "lcStrategic",
                                 "lcCooperative", "lcSpeedGain", "lcKeepRight"]
        self.assertCountEqual(attributes_1, expected_attributes_1)

        # test for SL2015
        lc_params_2 = SumoLaneChangeParams(model="SL2015")
        attributes_2 = list(lc_params_2.controller_params.keys())
        expected_attributes_2 = \
            ["laneChangeModel", "lcStrategic", "lcCooperative", "lcSpeedGain",
             "lcKeepRight", "lcLookaheadLeft", "lcSpeedGainRight", "lcSublane",
             "lcPushy", "lcPushyGap", "lcAssertive", "lcImpatience",
             "lcTimeToImpatience", "lcAccelLat"]
        self.assertCountEqual(attributes_2, expected_attributes_2)


if __name__ == '__main__':
    unittest.main()
