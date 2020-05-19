from gym import Env
from gym.spaces import Box
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env
from flow.envs.multiagent.base import MultiEnv
from ray.rllib.env import MultiAgentEnv

class DummyEnv(MultiAgentEnv):
    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(20, ))
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(2,))
    def step(self, action):
        return {'test': np.zeros(self.observation_space.shape)}, {'test': 1}, {'test': False, '__all__': False}, {}
    def reset(self):
        return {'test': np.zeros(self.observation_space.shape)}

def env_creator(env_config):
    return DummyEnv()
    
if __name__=='__main__':
    num_cpus = 2
    horizon = 2000

    ray.init(object_store_memory=int(1e9), redis_max_memory=int(1e9), memory=int(1e9))
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    register_env('DummyEnv', env_creator)
    config['env'] = 'CrowdSim'
    config['num_workers'] = num_cpus
    config['lr'] = tune.grid_search([1e-3, 1e-4, 1e-5, 1e-6])
    config["horizon"] = horizon
    config['train_batch_size'] = horizon * num_cpus
    
    temp_env = DummyEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    policy_graphs = {'av': (None, obs_space, act_space, {})}
    policies_to_train = ['av']
    def policy_mapping_fn(_):
        """Map a policy in RLlib."""
        return 'av'

    config['multiagent'].update({'policies': policy_graphs})
    config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
    config['multiagent'].update({'policies_to_train': policies_to_train})

    register_env('CrowdSim', env_creator)
    exp_dict = {
        'name': 'Test',
        'run_or_experiment': alg_run,
        'checkpoint_freq': 1000,
        'stop': {
            'training_iteration': 10000
        },
        'config': config,
    }
    run_tune(**exp_dict, queue_trials=False)