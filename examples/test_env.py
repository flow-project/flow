'''Use this to construct envs for debugging'''
import sys

from train import parse_args
from flow.utils.registry import make_create_env

def main(args):
  """Perform the training operations."""
  # Parse script-level arguments (not including package arguments).
  flags = parse_args(args)

  # Import relevant information from the exp_config script.
  module = __import__(
    "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
  module_ma = __import__(
    "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

  # Import the sub-module containing the specified exp_config and determine
  # whether the environment is single agent or multi-agent.
  if hasattr(module, flags.exp_config):
    submodule = getattr(module, flags.exp_config)
    multiagent = False
  elif hasattr(module_ma, flags.exp_config):
    submodule = getattr(module_ma, flags.exp_config)
    assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
      "Currently, multiagent experiments are only supported through " \
      "RLlib. Try running this experiment using RLlib: " \
      "'python train.py EXP_CONFIG'"
    multiagent = True
  else:
    raise ValueError("Unable to find experiment config.")

  flow_params = submodule.flow_params
  create_env, gym_name = make_create_env(params=flow_params)
  env = create_env()
  env.reset()
  import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main(sys.argv[1:])