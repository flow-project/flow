"""A more stable successor to TD3.

By default, this uses a near-identical configuration to that reported in the
TD3 paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import logging
import collections

import ray
from flow.algorithms.buffers import PrioritizedReplayBufferWithExperts
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.compression import pack_if_needed
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.memory import ray_get_and_free

from ray.rllib.optimizers import SyncReplayOptimizer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.schedules import LinearSchedule

from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, \
    DEFAULT_CONFIG as DDPG_CONFIG
from ray.rllib.utils import merge_dicts

logger = logging.getLogger(__name__)

TD3_DEFAULT_CONFIG = merge_dicts(
    DDPG_CONFIG,
    {
        # largest changes: twin Q functions, delayed policy updates, and target
        # smoothing
        "twin_q": True,
        "policy_delay": 2,
        "smooth_target_policy": True,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,

        # other changes & things we want to keep fixed: IID Gaussian
        # exploration noise, larger actor learning rate, no l2 regularisation,
        # no Huber loss, etc.
        "exploration_should_anneal": False,
        "exploration_noise_type": "gaussian",
        "exploration_gaussian_sigma": 0.1,
        "learning_starts": 10000,
        "pure_exploration_steps": 10000,
        "actor_hiddens": [400, 300],
        "critic_hiddens": [400, 300],
        "n_step": 1,
        "gamma": 0.99,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "l2_reg": 0.0,
        "tau": 5e-3,
        "train_batch_size": 100,
        "use_huber": False,
        "target_network_update_freq": 0,
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "per_worker_exploration": False,
        "worker_side_prioritization": False,
        "buffer_size": 1000000,
        "prioritized_replay": False,
        "clip_rewards": False,
        "use_state_preprocessor": False,
    },
)


class DQfDOptimizer(SyncReplayOptimizer):
    """Variant of the local sync optimizer that supports replay (for DQN) and selects out expert actions for
    a fixed number of iterations.
    This optimizer requires that rollout workers return an additional
    "td_error" array in the info return of compute_gradients(). This error
    term will be used for sample prioritization."""

    def __init__(self,
                 workers,
                 learning_starts=1000,
                 buffer_size=10000,
                 prioritized_replay=True,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 schedule_max_timesteps=100000,
                 beta_annealing_fraction=0.2,
                 final_prioritized_replay_beta=0.4,
                 train_batch_size=32,
                 num_expert_steps=1e2,
                 sample_batch_size=4,
                 before_learn_on_batch=None,
                 synchronize_sampling=False,
                 reserved_frac=0.1):
        """Initialize an sync replay optimizer.
        Arguments:
            workers (WorkerSet): all workers
            learning_starts (int): wait until this many steps have been sampled
                before starting optimization.
            buffer_size (int): max size of the replay buffer
            prioritized_replay (bool): whether to enable prioritized replay
            prioritized_replay_alpha (float): replay alpha hyperparameter
            prioritized_replay_beta (float): replay beta hyperparameter
            prioritized_replay_eps (float): replay eps hyperparameter
            schedule_max_timesteps (int): number of timesteps in the schedule
            beta_annealing_fraction (float): fraction of schedule to anneal
                beta over
            final_prioritized_replay_beta (float): final value of beta
            train_batch_size (int): size of batches to learn on
            num_expert_steps (float): how many training iterations to take using the expert in the env
            sample_batch_size (int): size of batches to sample from workers
            before_learn_on_batch (function): callback to run before passing
                the sampled batch to learn on
            synchronize_sampling (bool): whether to sample the experiences for
                all policies with the same indices (used in MADDPG).
            reserved_frac (float): the percentage of the buffer reserved for the expert
        """
        PolicyOptimizer.__init__(self, workers)

        self.replay_starts = learning_starts
        # linearly annealing beta used in Rainbow paper
        self.prioritized_replay_beta = LinearSchedule(
            schedule_timesteps=int(
                schedule_max_timesteps * beta_annealing_fraction),
            initial_p=prioritized_replay_beta,
            final_p=final_prioritized_replay_beta)
        self.prioritized_replay_eps = prioritized_replay_eps
        self.train_batch_size = train_batch_size
        self.before_learn_on_batch = before_learn_on_batch
        self.synchronize_sampling = synchronize_sampling

        # Stats
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.learner_stats = {}

        # Tracking how long we have trained for
        self.num_expert_steps = num_expert_steps

        # Set up replay buffer
        # TODO(@evinitsky) make this work without prioritized replay
        def new_buffer():
            return PrioritizedReplayBufferWithExperts(
                buffer_size, alpha=prioritized_replay_alpha, reserved_frac=reserved_frac)

        self.replay_buffers = collections.defaultdict(new_buffer)

        if buffer_size < self.replay_starts:
            logger.warning("buffer_size={} < replay_starts={}".format(
                buffer_size, self.replay_starts))

    @override(PolicyOptimizer)
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                batch = SampleBatch.concat_samples(
                    ray_get_and_free([
                        e.sample.remote()
                        for e in self.workers.remote_workers()
                    ]))
            else:
                batch = self.workers.local_worker().sample()

            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)


            # TODO(make sure to set the exploration rate correctly)
            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    # replace the actions with the expert actions
                    if self.num_steps_sampled < self.num_expert_steps:
                        self.replay_buffers[policy_id].add_to_reserved(
                            pack_if_needed(row["obs"]),
                            int(row["obs"][-1]),
                            row["rewards"],
                            pack_if_needed(row["new_obs"]),
                            row["dones"],
                            weight=None)
                    else:
                        self.replay_buffers[policy_id].add(
                            pack_if_needed(row["obs"]),
                            row["actions"],
                            row["rewards"],
                            pack_if_needed(row["new_obs"]),
                            row["dones"],
                            weight=None)

        if self.num_steps_sampled >= self.replay_starts:
            self._optimize()

        self.num_steps_sampled += batch.count


def make_optimizer(workers, config):
    return DQfDOptimizer(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        prioritized_replay=config["prioritized_replay"],
        prioritized_replay_alpha=config["prioritized_replay_alpha"],
        prioritized_replay_beta=config["prioritized_replay_beta"],
        schedule_max_timesteps=config["schedule_max_timesteps"],
        beta_annealing_fraction=config["beta_annealing_fraction"],
        final_prioritized_replay_beta=config["final_prioritized_replay_beta"],
        prioritized_replay_eps=config["prioritized_replay_eps"],
        train_batch_size=config["train_batch_size"],
        num_expert_steps=config["num_expert_steps"],
        sample_batch_size=config["sample_batch_size"],
        reserved_frac=config["reserved_frac"],
        **config["optimizer"])


new_config = deepcopy(TD3_DEFAULT_CONFIG)
new_config["num_expert_steps"] = 5e4
new_config["reserved_frac"] = .1


TD3Trainer = DDPGTrainer.with_updates(
    name="TD3", default_config=TD3_DEFAULT_CONFIG)
