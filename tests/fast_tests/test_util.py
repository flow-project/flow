import unittest
import csv
import os
os.environ["TEST_FLAG"] = "True"

from flow.core.util import emission_to_csv


class TestEmissionToCSV(unittest.TestCase):
    """
    Tests the emission_to_csv function on a small file. Ensures that the headers
    are correct, the length is correct, and some of the components are correct.
    """

    def runTest(self):
        # current path
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]

        # run the emission_to_csv function on a small emission file
        emission_to_csv(current_path + "/test_files/test-emission.xml")

        # import the generated csv file and its headers
        dict1 = []
        with open(current_path + "/test_files/test-emission.csv", "r") as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            for row in reader:
                dict1.append(dict())
                for i, key in enumerate(headers):
                    dict1[-1][key] = row[i]

        # check the names of the headers
        expected_headers = \
            ['time', 'CO', 'y', 'CO2', 'electricity', 'type', 'id', 'eclass',
             'waiting', 'NOx', 'fuel', 'HC', 'x', 'route', 'relative_position',
             'noise', 'angle', 'PMx', 'speed', 'edge_id', 'lane_number']

        self.assertCountEqual(headers, expected_headers)

        # check the number of rows of the generated csv file
        # Note that, rl vehicles are missing their final (reset) values, which I
        # don't think is a problem
        self.assertEqual(len(dict1), 104)


if __name__ == '__main__':
    unittest.main()
