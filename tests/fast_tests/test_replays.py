from flow.replay import rllib_replay as rl_replay
from flow.replay.rllib_replay import replay_rllib

import os
import unittest
import ray

os.environ['TEST_FLAG'] = 'True'


class TestRLReplay(unittest.TestCase):
    """Tests rllib_replay"""

    def test_rllib_replay_single(self):
        """Test for single agent replay"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/single_agent 1 --num_rollouts 1 ' \
                  '--render_mode no_render ' \
                  '--horizon 10'.format(current_path).split()
        parser = rl_replay.create_parser()
        pass_args = parser.parse_args(arg_str)
        replay_rllib(pass_args)

    # FIXME(ev) set the horizon so that this runs faster
    def test_rllib_replay_multi(self):
        """Test for multi-agent replay"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/multi_agent 1 --num_rollouts 1 ' \
                  '--render_mode no_render ' \
                  '--horizon 10'.format(current_path).split()
        parser = rl_replay.create_parser()
        pass_args = parser.parse_args(arg_str)
        replay_rllib(pass_args)


if __name__ == '__main__':
    ray.init(num_cpus=1)
    unittest.main()
    ray.shutdown()
