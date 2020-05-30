from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""An example of customizing PPO to leverage a centralized critic."""

import argparse
import numpy as np

from gym.spaces import Dict

from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from flow.algorithms.custom_ppo import CustomPPOTFPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.utils.annotations import override
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils import try_import_tf


tf = try_import_tf()

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"

CENTRAL_OBS = "central_obs"
OPPONENT_ACTION = "opponent_action"

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=100000)

#TODOy

class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized VF."""
    # TODO(@evinitsky) make this work with more than boxes

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        # Base of the model
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

        # Central VF maps (obs, opp_ops, opp_act) -> vf_pred
        self.max_num_agents = model_config['custom_options']['max_num_agents']
        self.obs_space_shape = obs_space.shape[0]
        self.obs_space = obs_space
        other_obs = tf.keras.layers.Input(shape=(obs_space.shape[0] * self.max_num_agents, ), name="central_obs")
        central_vf_dense = tf.keras.layers.Dense(
            model_config['custom_options']['central_vf_size'], activation=tf.nn.tanh, name="c_vf_dense")(other_obs)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense)
        self.central_vf = tf.keras.Model(
            inputs=[other_obs], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, central_obs):
        return tf.reshape(
            self.central_vf(
                [central_obs]), [-1])

    def value_function(self):
        return self.model.value_function()  # not used


# TODO(@evinitsky) support recurrence
class CentralizedCriticModelRNN(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=64,
                 cell_size=64):
        super(CentralizedCriticModelRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.model.variables)
        self.model.summary()

        #TODO(@evinitsky) add layer sharing to the VF
        # Create the centralized VF
        # Central VF maps (obs, opp_ops, opp_act) -> vf_pred
        self.max_num_agents = model_config.get("max_num_agents", 120)
        self.obs_space_shape = obs_space.shape[0]
        other_obs = tf.keras.layers.Input(shape=(obs_space.shape[0] * self.max_num_agents,), name="all_agent_obs")
        central_vf_dense = tf.keras.layers.Dense(
            model_config.get("central_vf_size", 64), activation=tf.nn.tanh, name="c_vf_dense")(other_obs)
        central_vf_dense2 = tf.keras.layers.Dense(
            model_config.get("central_vf_size", 64), activation=tf.nn.tanh, name="c_vf_dense")(central_vf_dense)
        central_vf_out = tf.keras.layers.Dense(
            1, activation=None, name="c_vf_out")(central_vf_dense2)
        self.central_vf = tf.keras.Model(
            inputs=[other_obs], outputs=central_vf_out)
        self.register_variables(self.central_vf.variables)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def central_value_function(self, central_obs):
        return tf.reshape(
            self.central_vf(
                [central_obs]), [-1])

    def value_function(self):
        return tf.reshape(self._value_out, [-1])  # not used


class CentralizedValueMixin(object):
    """Add methods to evaluate the central value function from the model."""

    def __init__(self):
        # TODO(@evinitsky) clean up naming
        self.central_value_function = self.model.central_value_function(
            self.get_placeholder(CENTRAL_OBS)
        )

    def compute_central_vf(self, central_obs):
        feed_dict = {
            self.get_placeholder(CENTRAL_OBS): central_obs,
        }
        return self.get_session().run(self.central_value_function, feed_dict)


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    if policy.loss_initialized():
        assert other_agent_batches is not None

        # time_span = (sample_batch['t'][0], sample_batch['t'][-1])
        # # there's a new problem here, namely that a segment might not be continuous due to the rerouting
        # other_agent_timespans = {agent_id:
        #                          (other_agent_batches[agent_id][1]["t"][0],
        #                           other_agent_batches[agent_id][1]["t"][-1])
        #                      for agent_id in other_agent_batches.keys()}
        other_agent_times = {agent_id: other_agent_batches[agent_id][1]["t"]
                             for agent_id in other_agent_batches.keys()}
        agent_time = sample_batch['t']
        # # find agents whose time overlaps with the current agent
        rel_agents = {agent_id: other_agent_time for agent_id, other_agent_time in other_agent_times.items()}
        # if len(rel_agents) > 0:
        other_obs = {agent_id:
                         other_agent_batches[agent_id][1]["obs"].copy()
                     for agent_id in other_agent_batches.keys()}
        # padded_agent_obs = {agent_id:
        #     overlap_and_pad_agent(
        #         time_span,
        #         rel_agent_time,
        #         other_obs[agent_id])
        #     for agent_id,
        #         rel_agent_time in rel_agents.items()}
        padded_agent_obs = {agent_id:
            fill_missing(
                agent_time,
                other_agent_times[agent_id],
                other_obs[agent_id])
            for agent_id,
                rel_agent_time in rel_agents.items()}
        # okay, now we need to stack and sort
        central_obs_list = [padded_obs for padded_obs in padded_agent_obs.values()]
        try:
            central_obs_batch = np.hstack((sample_batch["obs"], np.hstack(central_obs_list)))
        except:
            # TODO(@ev) this is a bug and needs to be fixed
            central_obs_batch = sample_batch["obs"]
        max_vf_agents = policy.model.max_num_agents
        num_agents = len(rel_agents) + 1
        if num_agents < max_vf_agents:
            diff = max_vf_agents - num_agents
            zero_pad = np.zeros((central_obs_batch.shape[0],
                                 policy.model.obs_space_shape * diff))
            central_obs_batch = np.hstack((central_obs_batch,
                                           zero_pad))
        elif num_agents > max_vf_agents:
            print("Too many agents!")

        # also record the opponent obs and actions in the trajectory
        sample_batch[CENTRAL_OBS] = central_obs_batch

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(sample_batch[CENTRAL_OBS])
    else:
        # policy hasn't initialized yet, use zeros
        #TODO(evinitsky) put in the right shape
        obs_shape = sample_batch[SampleBatch.CUR_OBS].shape[1]
        obs_shape = (1, obs_shape * (policy.model.max_num_agents))
        sample_batch[CENTRAL_OBS] = np.zeros(obs_shape)
        # TODO(evinitsky) put in the right shape. Will break if actions aren't 1
        sample_batch[SampleBatch.VF_PREDS] = np.zeros(1, dtype=np.float32)

    completed = sample_batch["dones"][-1]

    # if not completed and policy.loss_initialized():
    #     last_r = 0.0
    # else:
    #     next_state = []
    #     for i in range(policy.num_state_tensors()):
    #         next_state.append([sample_batch["state_out_{}".format(i)][-1]])
    #     last_r = policy.compute_central_vf(sample_batch[CENTRAL_OBS][-1][np.newaxis, ...])[0]

    batch = compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch



def time_overlap(time_span, agent_time):
    """Check if agent_time overlaps with time_span"""
    if agent_time[0] <= time_span[1] and agent_time[1] >= time_span[0]:
        return True
    else:
        return False


def fill_missing(agent_time, other_agent_time, obs):
    # shortcut, the two overlap perfectly
    if np.sum(agent_time == other_agent_time) == agent_time.shape[0]:
        return obs
    new_obs = np.zeros((agent_time.shape[0], obs.shape[1]))
    other_agent_time_set = set(other_agent_time)
    for i, time in enumerate(agent_time):
        if time in other_agent_time_set:
            new_obs[i] = obs[np.where(other_agent_time == time)]
    return new_obs


def overlap_and_pad_agent(time_span, agent_time, obs):
    """take the part of obs that overlaps, pad to length time_span
    Arguments:
        time_span (tuple): tuple of the first and last time that the agent
            of interest is in the system
        agent_time (tuple): tuple of the first and last time that the
            agent whose obs we are padding is in the system
        obs (np.ndarray): observations of the agent whose time is
            agent_time
    """
    assert time_overlap(time_span, agent_time)
    print(time_span)
    print(agent_time)
    if time_span[0] == 7 or agent_time[0] == 7:
        import ipdb; ipdb.set_trace()
    # FIXME(ev) some of these conditions can be combined
    # no padding needed
    if agent_time[0] == time_span[0] and agent_time[1] == time_span[1]:
        if obs.shape[0] < 200:
            import ipdb; ipdb.set_trace()
        return obs
    # agent enters before time_span starts and exits before time_span end
    if agent_time[0] < time_span[0] and agent_time[1] < time_span[1]:
        non_overlap_time = time_span[0] - agent_time[0]
        missing_time = time_span[1] - agent_time[1]
        overlap_obs = obs[non_overlap_time:]
        padding = np.zeros((missing_time, obs.shape[1]))
        obs_concat = np.concatenate((overlap_obs, padding))
        if obs_concat.shape[0] < 200:
            import ipdb; ipdb.set_trace()
        return obs_concat
    # agent enters after time_span starts and exits after time_span ends
    elif agent_time[0] > time_span[0] and agent_time[1] > time_span[1]:
        non_overlap_time = agent_time[1] - time_span[1]
        overlap_obs = obs[:-non_overlap_time]
        missing_time = agent_time[0] - time_span[0]
        padding = np.zeros((missing_time, obs.shape[1]))
        obs_concat = np.concatenate((padding, overlap_obs))
        if obs_concat.shape[0] < 200:
            import ipdb; ipdb.set_trace()
        return obs_concat
    # agent time is entirely contained in time_span
    elif agent_time[0] >= time_span[0] and agent_time[1] <= time_span[1]:
        missing_left = agent_time[0] - time_span[0]
        missing_right = time_span[1] - agent_time[1]
        obs_concat = obs
        if missing_left > 0:
            padding = np.zeros((missing_left, obs.shape[1]))
            obs_concat = np.concatenate((padding, obs_concat))
        if missing_right > 0:
            padding = np.zeros((missing_right, obs.shape[1]))
            obs_concat = np.concatenate((obs_concat, padding))
        if obs_concat.shape[0] < 200:
            import ipdb; ipdb.set_trace()
        return obs_concat
    # agent time totally contains time_span
    elif agent_time[0] <= time_span[0] and agent_time[1] >= time_span[1]:
        non_overlap_left = time_span[0] - agent_time[0]
        non_overlap_right = agent_time[1] - time_span[1]
        overlap_obs = obs
        if non_overlap_left > 0:
            overlap_obs = overlap_obs[non_overlap_left:]
        if non_overlap_right > 0:
            overlap_obs = overlap_obs[:-non_overlap_right]
        if overlap_obs.shape[0] < 200:
            import ipdb; ipdb.set_trace()
        return overlap_obs


# Copied from PPO but optimizing the central value function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_function,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss


class PPOLoss(object):
    def __init__(self,
                 action_space,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True,
                 model_config=None):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
            model_config (dict): (Optional) model config for use in specifying
                action distributions.
        """

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    loss = loss_with_central_critic(policy, model, dist_class, train_batch)
    return loss


class KLCoeffMixin(object):
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)
    def update_kl(self, blah):
        pass


def setup_mixins(policy, obs_space, action_space, config):
    # copied from PPO
    KLCoeffMixin.__init__(policy, config)

    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    # hack: put in a noop VF so some of the inherited PPO code runs
    policy.value_function = tf.zeros(
        tf.shape(policy.get_placeholder(SampleBatch.CUR_OBS))[0])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_function),
    }

def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "vf_preds": train_batch[Postprocessing.VALUE_TARGETS],
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }

CCPPO = CustomPPOTFPolicy.with_updates(
    name="CCPPO",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=new_ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule,
        CentralizedValueMixin, KLCoeffMixin
    ])

CCTrainer = PPOTrainer.with_updates(name="CCPPOTrainer", default_policy=CCPPO)