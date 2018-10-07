import unittest
import csv
import os
import json
import collections

from flow.core.vehicles import Vehicles
from flow.core.traffic_lights import TrafficLights
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoCarFollowingParams
from flow.core.util import emission_to_csv
from flow.utils.flow_warnings import deprecation_warning
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder, get_flow_params

os.environ["TEST_FLAG"] = "True"


class TestEmissionToCSV(unittest.TestCase):
    """Tests the emission_to_csv function on a small file.

    Ensures that the headers are correct, the length is correct, and some of
    the components are correct.
    """

    def test_emission_to_csv(self):
        # current path
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]

        # run the emission_to_csv function on a small emission file
        emission_to_csv(current_path + "/test_files/test-emission.xml")

        # import the generated csv file and its headers
        dict1 = []
        filename = current_path + "/test_files/test-emission.csv"
        with open(filename, "r") as infile:
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
        # Note that, rl vehicles are missing their final (reset) values, which
        # I don't think is a problem
        self.assertEqual(len(dict1), 104)


class TestWarnings(unittest.TestCase):
    """Tests warning functions located in flow.utils.warnings"""

    def test_deprecation_warning(self):
        # dummy class
        class Foo(object):
            pass

        # dummy attribute name
        dep_from = "bar_deprecated"
        dep_to = "bar_new"

        # check the deprecation warning is printing what is expected
        self.assertWarnsRegex(
            UserWarning, "The attribute bar_deprecated in Foo is deprecated, "
            "use bar_new instead.", deprecation_warning, Foo(), dep_from,
            dep_to)


class TestRegistry(unittest.TestCase):
    """Tests the methods located in flow/utils/registry.py"""

    def test_make_create_env(self):
        """Tests that the make_create_env methods generates an environment with
        the expected flow parameters."""
        # use a flow_params dict derived from flow/benchmarks/figureeight0.py
        vehicles = Vehicles()
        vehicles.add(
            veh_id="human",
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=13)
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=1)

        flow_params = dict(
            exp_tag="figure_eight_0",
            env_name="AccelEnv",
            scenario="Figure8Scenario",
            generator="Figure8Generator",
            sumo=SumoParams(
                sim_step=0.1,
                render=False,
            ),
            env=EnvParams(
                horizon=1500,
                additional_params={
                    "target_velocity": 20,
                    "max_accel": 3,
                    "max_decel": 3,
                },
            ),
            net=NetParams(
                no_internal_links=False,
                additional_params={
                    "radius_ring": 30,
                    "lanes": 1,
                    "speed_limit": 30,
                    "resolution": 40,
                },
            ),
            veh=vehicles,
            initial=InitialConfig(),
            tls=TrafficLights(),
        )

        # some random version number for testing
        v = 23434

        # call make_create_env
        create_env, env_name = make_create_env(params=flow_params, version=v)

        # check that the name is correct
        self.assertEqual(env_name, '{}-v{}'.format(flow_params["env_name"], v))

        # create the gym environment
        env = create_env()

        # Note that we expect the port number in sumo_params to change, and
        # that this feature is in fact needed to avoid race conditions
        flow_params["sumo"].port = env.env.sumo_params.port

        # check that each of the parameter match
        self.assertEqual(env.env.env_params.__dict__,
                         flow_params["env"].__dict__)
        self.assertEqual(env.env.sumo_params.__dict__,
                         flow_params["sumo"].__dict__)
        self.assertEqual(env.env.traffic_lights.__dict__,
                         flow_params["tls"].__dict__)
        self.assertEqual(env.env.scenario.net_params.__dict__,
                         flow_params["net"].__dict__)
        self.assertEqual(env.env.scenario.net_params.__dict__,
                         flow_params["net"].__dict__)
        self.assertEqual(env.env.scenario.initial_config.__dict__,
                         flow_params["initial"].__dict__)
        self.assertEqual(env.env.__class__.__name__, flow_params["env_name"])
        self.assertEqual(env.env.scenario.__class__.__name__,
                         flow_params["scenario"])
        self.assertEqual(env.env.scenario.generator_class.__name__,
                         flow_params["generator"])


class TestRllib(unittest.TestCase):
    """Tests the methods located in flow/utils/rllib.py"""

    def test_encoder_and_get_flow_params(self):
        """Tests both FlowParamsEncoder and get_flow_params.

        FlowParamsEncoder is used to serialize the data from a flow_params dict
        for replay by the visualizer later. Then, the get_flow_params method is
        used to try and read the parameters from the config file, and is
        checked to match expected results.
        """
        # use a flow_params dict derived from flow/benchmarks/merge0.py
        vehicles = Vehicles()
        vehicles.add(
            veh_id="human",
            acceleration_controller=(IDMController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=5)
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                speed_mode="no_collide",
            ),
            num_vehicles=0)

        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="inflow_highway",
            vehs_per_hour=1800,
            departLane="free",
            departSpeed=10)
        inflow.add(
            veh_type="rl",
            edge="inflow_highway",
            vehs_per_hour=200,
            departLane="free",
            departSpeed=10)
        inflow.add(
            veh_type="human",
            edge="inflow_merge",
            vehs_per_hour=100,
            departLane="free",
            departSpeed=7.5)

        flow_params = dict(
            exp_tag="merge_0",
            env_name="WaveAttenuationMergePOEnv",
            scenario="MergeScenario",
            generator="MergeGenerator",
            sumo=SumoParams(
                restart_instance=True,
                sim_step=0.5,
                render=False,
            ),
            env=EnvParams(
                horizon=750,
                sims_per_step=2,
                warmup_steps=0,
                additional_params={
                    "max_accel": 1.5,
                    "max_decel": 1.5,
                    "target_velocity": 20,
                    "num_rl": 5,
                },
            ),
            net=NetParams(
                inflows=inflow,
                no_internal_links=False,
                additional_params={
                    "merge_length": 100,
                    "pre_merge_length": 500,
                    "post_merge_length": 100,
                    "merge_lanes": 1,
                    "highway_lanes": 1,
                    "speed_limit": 30,
                },
            ),
            veh=vehicles,
            initial=InitialConfig(),
            tls=TrafficLights(),
        )

        # create an config dict with space for the flow_params dict
        config = {"env_config": {}}

        # save the flow params for replay
        flow_json = json.dumps(
            flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
        config['env_config']['flow_params'] = flow_json

        # dump the config so we can fetch it
        json_out_file = 'params.json'
        with open(os.path.expanduser(json_out_file), 'w+') as outfile:
            json.dump(
                config,
                outfile,
                cls=FlowParamsEncoder,
                sort_keys=True,
                indent=4)

        # fetch values using utility function `get_flow_params`
        imported_flow_params = get_flow_params(config)

        # delete the created file
        os.remove(os.path.expanduser('params.json'))

        # test that this inflows are correct
        self.assertTrue(imported_flow_params["net"].inflows.__dict__ ==
                        flow_params["net"].inflows.__dict__)

        imported_flow_params["net"].inflows = None
        flow_params["net"].inflows = None

        # make sure the rest of the imported flow_params match the originals
        self.assertTrue(imported_flow_params["env"].__dict__ == flow_params[
            "env"].__dict__)
        self.assertTrue(imported_flow_params["initial"].__dict__ ==
                        flow_params["initial"].__dict__)
        self.assertTrue(imported_flow_params["tls"].__dict__ == flow_params[
            "tls"].__dict__)
        self.assertTrue(imported_flow_params["sumo"].__dict__ == flow_params[
            "sumo"].__dict__)
        self.assertTrue(imported_flow_params["net"].__dict__ == flow_params[
            "net"].__dict__)

        self.assertTrue(
            imported_flow_params["exp_tag"] == flow_params["exp_tag"])
        self.assertTrue(
            imported_flow_params["env_name"] == flow_params["env_name"])
        self.assertTrue(
            imported_flow_params["scenario"] == flow_params["scenario"])
        self.assertTrue(
            imported_flow_params["generator"] == flow_params["generator"])

        def search_dicts(obj1, obj2):
            """Searches through dictionaries as well as lists of dictionaries
            recursively to determine if any two components are mismatched."""
            for key in obj1.keys():
                # if an next element is a list, either compare the two lists,
                # or if the lists contain dictionaries themselves, look at each
                # dictionary component recursively to check for mismatches
                if isinstance(obj1[key], list):
                    if len(obj1[key]) > 0:
                        if isinstance(obj1[key][0], dict):
                            for i in range(len(obj1[key])):
                                if not search_dicts(obj1[key][i],
                                                    obj2[key][i]):
                                    return False
                        elif obj1[key] != obj2[key]:
                            return False
                # if the next element is a dict, run through it recursively to
                # determine if the separate elements of the dict match
                if isinstance(obj1[key], (dict, collections.OrderedDict)):
                    if not search_dicts(obj1[key], obj2[key]):
                        return False
                # if it is neither a list or a dictionary, compare to determine
                # if the two elements match
                elif obj1[key] != obj2[key]:
                    # if the two elements that are being compared are objects,
                    # make sure that they are the same type
                    if not isinstance(obj1[key], type(obj2[key])):
                        return False
            return True

        # make sure that the Vehicles class that was imported matches the
        # original one
        if not search_dicts(imported_flow_params["veh"].__dict__,
                            flow_params["veh"].__dict__):
            raise AssertionError


if __name__ == '__main__':
    unittest.main()
