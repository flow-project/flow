"""Runner script for RLlib experiments with macroscopic models.

To run an experiment with this script, type:

    python train_rllib.py MODEL_NAME

where, model name is one of: {"ARZ", "LWR", "NonLocal"}.

The network parameters for each model can also be passed as additional
arguments through the command line. For example, when using the LWR model, the
following is valid:

    python train_rllib.py "LWR" --length 10000 --dx 100

Other, common, trainable arguments include:
  - n_itr : number of training epochs before the operation is exited
  - n_rollouts: number of rollouts per training iteration
"""


def run_training(n_itr, n_rollouts, env_name, env_params, net_params):
    """

    Parameters
    ----------
    n_itr : int
        number of training epochs before the operation is exited
    n_rollouts : int
        number of rollouts per training iteration
    """
    pass


def rollout():
    pass  # FIXME


if __name__ == "__main__":
    pass
