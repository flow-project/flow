"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import sys

import ray

from flow.core.experiment import Experiment


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')

    # optional input parameters
    parser.add_argument(
        '--num_runs', type=int, default=1,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--aimsun',
        action='store_true',
        help='Specifies whether to run the simulation using the simulator '
             'Aimsun. If not specified, the simulator used is SUMO.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    return parser.parse_known_args(args)[0]

@ray.remote
def run_experiment(flow_params, custom_callables):
    exp = Experiment(flow_params, custom_callables)
    info_dict = exp.run(flags.num_runs, convert_to_csv=flags.gen_emission)
    return info_dict


if __name__ == "__main__":
    ray.init(local_mode=True)
    flags = parse_args(sys.argv[1:])

    # Get the flow_params object.
    module = __import__("exp_configs.non_rl", fromlist=[flags.exp_config])
    flow_params_list = getattr(module, flags.exp_config).flow_params

    # Get the custom callables for the runner
    custom_callables = []
    if hasattr(getattr(module, flags.exp_config), "custom_callables"):
        custom_callables = getattr(module, flags.exp_config).custom_callables

    # Update some variables based on inputs.
    for flow_params in flow_params_list:
        flow_params['sim'].render = not flags.no_render
        flow_params['simulator'] = 'aimsun' if flags.aimsun else 'traci'

        # specify an emission path if they are meant to be generated
        if flags.gen_emission:
            flow_params['sim'].emission_path = "./data"

    temp_output = [run_experiment.remote(flow_params=flow_params,
                                         custom_callables=custom_callables) for flow_params in flow_params_list]
    temp_output = ray.get(temp_output)
