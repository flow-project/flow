"""
Utility functions for Flow compatibility with the softlearning library (SAC).

This includes:
- ExperimentRunner class
- get_variant_spec function
- generate_experiment_kwargs function
- adapt_environment_for_sac function
"""

import os
import copy
import glob
import pickle
import types
import json
from collections import defaultdict
import numpy as np

import tensorflow as tf
from ray import tune

from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import set_seed, initialize_tf_variables
from softlearning.misc.utils import datetimestamp

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


def get_variant_spec(params):
    variant_spec = {
        'env_config': {
            'flow_params': json.dumps(params['flow_params'],
                                      cls=FlowParamsEncoder,
                                      sort_keys=True, indent=4)
        },
        'flow_params': params['flow_params'],
        'environment_params': {
            'training': {
                'kwargs': {
                    # Add environment params here
                },
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': params['gaussian_policy_params'],
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': params['gaussian_policy_params']
                                            ['kwargs']['hidden_layer_sizes'],
            }
        },
        'algorithm_params': params['algorithm_params'],
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': params['sampler_params'],
        'run_params': params['run_params'],
        'resources_per_trial': params['resources_per_trial']
    }

    return variant_spec


def generate_experiment_kwargs(variant_spec):
    exp_name = variant_spec['flow_params']['exp_tag']
    experiment_id = '-'.join((datetimestamp(), exp_name))
    local_dir = '~/ray_results/' + exp_name

    def create_trial_name_creator(trial_name_template):
        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)
        return tune.function(trial_name_creator)

    experiment_kwargs = {
        'name': experiment_id,
        'resources_per_trial': variant_spec['resources_per_trial'],
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': 1,
        'upload_dir': '',
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'trial_name_creator': create_trial_name_creator(
            'id={trial.trial_id}-seed={trial.config[run_params][seed]}'),
        'restore': None,
    }

    return experiment_kwargs


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        variant = copy.deepcopy(self._variant)

        # create Flow environment
        flow_params = variant['flow_params']
        create_env, _ = make_create_env(params=flow_params, version=0)
        env = create_env()

        adapt_environment_for_sac(env)

        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            env)
        evaluation_environment = self.evaluation_environment = (
            env
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    @property
    def picklables(self):
        env_params = self._variant['env_config']['flow_params']
        return {
            'variant': self._variant,
            'training_environment': env_params,
            'evaluation_environment': env_params,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.picklables, f)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        tf_checkpoint = self._get_tf_checkpoint()

        tf_checkpoint.save(
            file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),
            session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        flow_params = self._variant['flow_params']

        create_env, _ = make_create_env(params=flow_params, version=0)
        env = create_env()
        adapt_environment_for_sac(env)

        training_environment = self.training_environment = env
        evaluation_environment = self.evaluation_environment = env

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True


def adapt_environment_for_sac(env):
    # add necessary methods and attributes to the environment
    env.active_observation_shape = env.observation_space.shape
    env.convert_to_active_observation = types.MethodType(
        _convert_to_active_observation, env)
    env.get_path_infos = types.MethodType(_get_path_infos, env)


def _convert_to_active_observation(self, observation):
    return observation


def _get_path_infos(self, paths, *args, **kwargs):
    """Log some general diagnostics from the env infos.
    TODO(hartikainen): These logs don't make much sense right now. Need to
    figure out better format for logging general env infos.
    """
    keys = list(paths[0].get('infos', [{}])[0].keys())

    results = defaultdict(list)

    for path in paths:
        path_results = {
            k: [
                info[k]
                for info in path['infos']
            ] for k in keys
        }
        for info_key, info_values in path_results.items():
            info_values = np.array(info_values)
            results[info_key + '-first'].append(info_values[0])
            results[info_key + '-last'].append(info_values[-1])
            results[info_key + '-mean'].append(np.mean(info_values))
            results[info_key + '-median'].append(np.median(info_values))
            if np.array(info_values).dtype != np.dtype('bool'):
                results[info_key + '-range'].append(np.ptp(info_values))

    aggregated_results = {}
    for key, value in results.items():
        aggregated_results[key + '-mean'] = np.mean(value)

    return aggregated_results
