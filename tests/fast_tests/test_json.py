"""
Unit tests for JSON file output and reading (used for visualization)
"""

import json
import os
import unittest

import ray

from examples.rllib.stabilizing_the_ring import make_create_env

from flow.scenarios.loop.loop_scenario import LoopScenario
from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.util import NameEncoder, get_flow_params, get_rllib_config
import ray.rllib.ppo as ppo

os.environ["TEST_FLAG"] = "True"


class TestJSON(unittest.TestCase):
    # def setUp(self):
    #     # reload modules, required upon repeated ray.init()

    def test_json(self):
        """
        Integration test for json export and import workflow
        """
        horizon = 500

        additional_env_params = {"target_velocity": 8, "max-deacc": -1,
                                 "max-acc": 1, "num_steps": horizon}
        additional_net_params = {"length": 260, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        vehicle_params = [dict(veh_id="rl", num_vehicles=1,
                               acceleration_controller=(RLController, {}),
                               routing_controller=(ContinuousRouter, {})),
                          dict(veh_id="idm", num_vehicles=21,
                               acceleration_controller=(IDMController, {}),
                               routing_controller=(ContinuousRouter, {}))
                          ]

        flow_params = dict(
            sumo=dict(sim_step=0.1),
            env=dict(additional_params=additional_env_params),
            net=dict(no_internal_links=False,
                     additional_params=additional_net_params),
            veh=vehicle_params,
            initial=dict(spacing="uniform", bunching=30, min_gap=0)
        )

        flow_env_name = "WaveAttenuationPOEnv"
        exp_tag = "stabilizing_the_ring_example"  # experiment prefix

        flow_params['flowenv'] = flow_env_name
        flow_params['exp_tag'] = exp_tag
        flow_params['module'] = "stabilizing_the_ring"  # an example

        # Just to make sure an env can be created successfully
        # using these current flow_params
        create_env, env_name = make_create_env(flow_env_name, flow_params,
                                               version=0,
                                               exp_tag=exp_tag)

        config = ppo.DEFAULT_CONFIG.copy()
        # save the flow params for replay
        flow_json = json.dumps(flow_params, cls=NameEncoder, sort_keys=True,
                               indent=4)
        config['env_config']['flow_params'] = flow_json

        # dump the config so we can fetch it
        json_out_file = '~/params.json'
        with open(os.path.expanduser(json_out_file), 'w+') as outfile:
            json.dump(config, outfile, cls=NameEncoder, sort_keys=True, indent=4)

        config = get_rllib_config(os.path.expanduser('~'))

        # Fetching values using utility function `get_flow_params`
        imported_flow_params, mce = \
            get_flow_params(config)

        # Making sure the right make_create_env is returned
        self.assertTrue(mce is make_create_env)

        # Making sure the imported flow_params match the originals
        self.assertTrue(imported_flow_params == flow_params)

        # delete the created file
        os.remove(os.path.expanduser('~/params.json'))

    def tearDown(self):
        ray.worker.cleanup()


if __name__ == '__main__':
    unittest.main()
