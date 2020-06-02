from ray import tune
from flow.controllers.imitation_learning.ppo_model import *
from ray.rllib.agents import ppo
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


class Imitation_PPO_Trainable(tune.Trainable):
    def _setup(self, config):
        env_name = config['env']
        # agent_cls = get_agent_class(config['env_config']['run'])
        self.trainer = ppo.PPOTrainer(env=env_name, config=config)
        print("\n\n\nPOLICY_NAME")
        policy_id = list(self.trainer.get_weights().keys())[0]
        print(policy_id)
        print("\n\n\n")
        self.trainer.import_model(config['model']['custom_options']['h5_load_path'], policy_id)
        print("here")

    def _train(self):
        print("TRAIN CALLED")
        # return self.trainer.train()
        return self.trainer.train()

    # def train(self):
    #     print("TRAIN CALLED")
    #     return self.trainer.train()

    def _save(self, tmp_checkpoint_dir):
        return self.trainer._save(tmp_checkpoint_dir)

    def _restore(self, checkpoint):
        self.trainer.restore(checkpoint)

    def _log_result(self, result):
        self.trainer._log_result(result)

    def _stop(self):
        self.trainer.stop()

    def _export_model(self, export_formats, export_dir):
        return self.trainer.export_model(export_formats, export_dir=export_dir)





