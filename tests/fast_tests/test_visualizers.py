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

    def test_visualizer(self):
        try:
            ray.init(num_cpus=1)  # , redis_address="localhost:6379")
        except Exception:
            pass

        # current path
        current_path = os.path.realpath(__file__).rsplit('/', 1)[0]

        # run the experiment and check it doesn't crash
        arg_str = '{}/../data/rllib_data/ 1 --num_rollouts 1 ' \
                  '--no_render'.format(current_path).split()
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
    ray.init(num_cpus=2)
    unittest.main()
