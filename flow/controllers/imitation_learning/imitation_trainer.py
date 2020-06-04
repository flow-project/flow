from ray import tune
from flow.controllers.imitation_learning.ppo_model import *
from ray.rllib.agents import ppo
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


class Imitation_PPO_Trainable(tune.Trainable):
    """
    Class to train PPO with imitation, with Tune. Extends Trainable.
    """

    def _setup(self, config):
        """
        Sets up trainable. See superclass definition.
        """

        env_name = config['env']
        # agent_cls = get_agent_class(config['env_config']['run'])
        self.trainer = ppo.PPOTrainer(env=env_name, config=config)
        policy_id = list(self.trainer.get_weights().keys())[0]
        self.trainer.import_model(config['model']['custom_options']['h5_load_path'], policy_id)

    def _train(self):
        """
        Executes one training iteration on trainer. See superclass definition.
        """
        print("TRAIN CALLED")
        # return self.trainer.train()
        return self.trainer.train()

    def _save(self, tmp_checkpoint_dir):
        """
        Saves trainer. See superclass definition.
        """
        return self.trainer._save(tmp_checkpoint_dir)

    def _restore(self, checkpoint):
        """
        Restores trainer from checkpoint. See superclass definition.
        """
        self.trainer.restore(checkpoint)

    def _log_result(self, result):
        """
        Logs results of trainer. See superclass definition.
        """
        self.trainer._log_result(result)

    def _stop(self):
        """
        Stops trainer. See superclass definition.
        """
        self.trainer.stop()

    def _export_model(self, export_formats, export_dir):
        """
        Exports trainer model. See superclass definition.
        """
        return self.trainer.export_model(export_formats, export_dir=export_dir)





