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


def run_training(n_itr, n_rollouts, alg, alg_params, env_name, env_params):
    """Perform a training operation.

    Parameters
    ----------
    n_itr : int
        number of training epochs before the operation is exited
    n_rollouts : int
        number of rollouts per training iteration
    alg : str
        name of the RLlib algorithm
    alg_params : dict
        algorithm specific features
    env_name : str
        name of the model/environment. Must be one of: {"ARZ", "LWR",
        "NonLocal"}
    env_params : dict
        environment-specific features. See the definition of the separate
        models for more.
    """
    pass


def rollout():
    """Fill in. todo."""
    pass  # FIXME


if __name__ == "__main__":
    pass
