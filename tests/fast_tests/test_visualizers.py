# from flow.visualize import visualizer_rllab as vs_rllab
# from flow.visualize.visualizer_rllab import visualizer_rllab
from flow.visualize import visualizer_rllib as vs_rllib
from flow.visualize.visualizer_rllib import visualizer_rllib

import os
import unittest
import ray

os.environ['TEST_FLAG'] = 'True'


class TestVisualizerRLlib(unittest.TestCase):
    """Tests visualizer_rllib"""

    def test_visualizer_single(self):
        """Test for single agent"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/single_agent 1 --num-rollouts 1 ' \
                  '--render-mode no-render ' \
                  '--horizon 10'.format(current_path).split()
        parser = vs_rllib.create_parser()
        pass_args = parser.parse_args(arg_str)
        visualizer_rllib(pass_args)

    # FIXME(ev) set the horizon so that this runs faster
    def test_visualizer_multi(self):
        """Test for multi-agent visualization"""
        try:
            ray.init(num_cpus=1)
        except Exception:
            pass
        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/multi_agent 1 --num-rollouts 1 ' \
                  '--render-mode no-render ' \
                  '--horizon 10'.format(current_path).split()
        parser = vs_rllib.create_parser()
        pass_args = parser.parse_args(arg_str)
        visualizer_rllib(pass_args)


# class TestVisualizerRLlab(unittest.TestCase):
#     """Tests visualizer_rllab"""
#
#     def test_visualizer(self):
#         # current path
#         current_path = os.path.realpath(__file__).rsplit('/', 1)[0]
#         arg_str = '{}/../data/rllab_data/itr_0.pkl --num_rollouts 1 ' \
#                   '--no_render'.format(current_path).split()
#         parser = vs_rllab.create_parser()
#         pass_args = parser.parse_args(arg_str)
#         visualizer_rllab(pass_args)


if __name__ == '__main__':
    ray.init(num_cpus=1)
    unittest.main()
