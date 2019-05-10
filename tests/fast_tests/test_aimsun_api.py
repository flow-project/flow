
import flow.config as config
import flow.utils.aimsun.constants
from flow.utils.aimsun.api import FlowAimsunAPI
from flow.utils.aimsun.struct import InfVeh
import unittest
import os
import subprocess
import numpy as np


class TestConstants(unittest.TestCase):
    """Tests for various aspect of the Aimsun constants in
    flow/utils/aimsun/constants.py."""

    def test_none_equal(self):
        """Verify that no two constants share the same value."""
        # get the names of all variables in the constants file
        var = [item for item in dir(flow.utils.aimsun.constants)
               if not item.startswith("__")]

        # get the variable values from the names
        variables = []
        module = __import__("flow.utils.aimsun.constants", fromlist=var)
        for v in var:
            variables.append(getattr(module, v))

        # makes sure than no two numbers are the same
        self.assertEqual(len(var), np.unique(variables).shape[0])


class TestStruct(unittest.TestCase):
    """Tests for the objects in flow/utils/aimsun/struct.py."""

    def test_inf_veh(self):
        """Verify that the InfVeh object contains the expected attributes."""
        expected_variables = [
            'CurrentPos',
            'distance2End',
            'xCurrentPos',
            'yCurrentPos',
            'zCurrentPos',
            'xCurrentPosBack',
            'yCurrentPosBack',
            'zCurrentPosBack',
            'CurrentSpeed',
            'TotalDistance',
            'SectionEntranceT',
            'CurrentStopTime',
            'stopped',
            'idSection',
            'segment',
            'numberLane',
            'idJunction',
            'idSectionFrom',
            'idLaneFrom',
            'idSectionTo',
            'idLaneTo'
        ]

        obj = InfVeh()

        for val in expected_variables:
            self.assertIn(val, obj.__dict__.keys())


class TestDummyAPI(unittest.TestCase):
    """Tests the functionality of FlowAimsunAPI.

    This is done by creating a server using the flow/tests/dummy_server.py
    script, which create a similar server and runs on the Aimsun conda env used
    while running actually Aimsun processes. This dummy server outputs dummy
    values for each API command, and is used simply to validate the
    functionality of the API-side of things.
    """

    def setUp(self):
        # start the server's process
        self.proc = subprocess.Popen([
            os.path.join(config.AIMSUN_SITEPACKAGES, "bin/python"),
            os.path.join(config.PROJECT_PATH, 'tests/dummy_server.py')])

        # create the FlowAimsunKernel object
        self.kernel_api = FlowAimsunAPI(port=9999)

    def tearDown(self):
        # kill the process
        self.proc.kill()

    def test_getter_methods(self):
        # test the get entered IDs method when the list is not empty
        entered_ids = self.kernel_api.get_entered_ids()
        self.assertListEqual(entered_ids, [1, 2, 3, 4, 5])

        # test the get entered IDs method when the list is empty
        entered_ids = self.kernel_api.get_entered_ids()
        self.assertEqual(len(entered_ids), 0)

        # test the get exited IDs method when the list is not empty
        exited_ids = self.kernel_api.get_exited_ids()
        self.assertListEqual(exited_ids, [6, 7, 8, 9, 10])

        # test the get exited IDs method when the list is empty
        exited_ids = self.kernel_api.get_exited_ids()
        self.assertEqual(len(exited_ids), 0)

        # test the get_vehicle_static_info method
        static_info = self.kernel_api.get_vehicle_static_info(veh_id=1)
        self.assertEqual(static_info.report, 1)
        self.assertEqual(static_info.idVeh, 2)
        self.assertEqual(static_info.type, 3)
        self.assertEqual(static_info.length, 4)
        self.assertEqual(static_info.width, 5)
        self.assertEqual(static_info.maxDesiredSpeed, 6)
        self.assertEqual(static_info.maxAcceleration, 7)
        self.assertEqual(static_info.normalDeceleration, 8)
        self.assertEqual(static_info.maxDeceleration, 9)
        self.assertEqual(static_info.speedAcceptance, 10)
        self.assertEqual(static_info.minDistanceVeh, 11)
        self.assertEqual(static_info.giveWayTime, 12)
        self.assertEqual(static_info.guidanceAcceptance, 13)
        self.assertEqual(static_info.enrouted, 14)
        self.assertEqual(static_info.equipped, 15)
        self.assertEqual(static_info.tracked, 16)
        self.assertFalse(static_info.keepfastLane)
        self.assertEqual(static_info.headwayMin, 18)
        self.assertEqual(static_info.sensitivityFactor, 19)
        self.assertEqual(static_info.reactionTime, 20)
        self.assertEqual(static_info.reactionTimeAtStop, 21)
        self.assertEqual(static_info.reactionTimeAtTrafficLight, 22)
        self.assertEqual(static_info.centroidOrigin, 23)
        self.assertEqual(static_info.centroidDest, 24)
        self.assertEqual(static_info.idsectionExit, 25)
        self.assertEqual(static_info.idLine, 26)

        tracking_inf = self.kernel_api.get_vehicle_tracking_info(
            veh_id=1, info_bitmap='1'*21)
        self.assertEqual(tracking_inf.CurrentPos, 4)
        self.assertEqual(tracking_inf.distance2End, 5)
        self.assertEqual(tracking_inf.xCurrentPos, 6)
        self.assertEqual(tracking_inf.yCurrentPos, 7)
        self.assertEqual(tracking_inf.zCurrentPos, 8)
        self.assertEqual(tracking_inf.xCurrentPosBack, 9)
        self.assertEqual(tracking_inf.yCurrentPosBack, 10)
        self.assertEqual(tracking_inf.zCurrentPosBack, 11)
        self.assertEqual(tracking_inf.CurrentSpeed, 12)
        self.assertEqual(tracking_inf.TotalDistance, 14)
        self.assertEqual(tracking_inf.SectionEntranceT, 17)
        self.assertEqual(tracking_inf.CurrentStopTime, 18)
        self.assertEqual(tracking_inf.stopped, 19)
        self.assertEqual(tracking_inf.idSection, 20)
        self.assertEqual(tracking_inf.segment, 21)
        self.assertEqual(tracking_inf.numberLane, 22)
        self.assertEqual(tracking_inf.idJunction, 23)
        self.assertEqual(tracking_inf.idSectionFrom, 24)
        self.assertEqual(tracking_inf.idLaneFrom, 25)
        self.assertEqual(tracking_inf.idSectionTo, 26)
        self.assertEqual(tracking_inf.idLaneTo, 27)

        # test the get traffic light IDs method when the list is not empty
        tl_ids = self.kernel_api.get_traffic_light_ids()
        self.assertListEqual(tl_ids, [1, 2, 3, 4, 5])

        # test the get traffic light IDs method when the list is empty
        tl_ids = self.kernel_api.get_traffic_light_ids()
        self.assertEqual(len(tl_ids), 0)


if __name__ == '__main__':
    unittest.main()
