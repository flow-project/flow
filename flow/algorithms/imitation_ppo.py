from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo_policy import LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import Policy
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats
import tensorflow as tf

BEHAVIOUR_LOGITS = "behaviour_logits"


def imitation_loss(policy, model, dist_class, train_batch):
    original_space = restore_original_dimensions(train_batch['obs'], model.obs_space)
    expert_tensor = original_space['expert_action']
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    if policy.config['model']['custom_options']["hard_negative_mining"]:
        # negative sign makes values that are really negative be selected in the top k
        masked_logp = -tf.boolean_mask(action_dist.logp(expert_tensor), mask)
        top_loss, _ = tf.math.top_k(masked_logp,
                                    int(policy.config['sgd_minibatch_size'] *
                                        policy.config['model']['custom_options']["mining_frac"]))
        imitation_loss = tf.reduce_mean(top_loss)

    else:
        # We want to maximize log likelihood, we flip the sign so that we are minimizing the negative log prob
        imitation_loss = -tf.reduce_mean(tf.boolean_mask(action_dist.logp(expert_tensor), mask))

    return imitation_loss


# def new_ppo_surrogate_loss(policy, batch_tensors):
def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    policy.imitation_loss = imitation_loss(policy, model, dist_class, train_batch)
    loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    return policy.policy_weight * loss + policy.imitation_weight * policy.imitation_loss


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
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

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
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    return policy.loss_obj.loss


@DeveloperAPI
class ImitationLearningRateSchedule(object):
    """Mixin for TFPolicy that adds a learning rate schedule."""

    @DeveloperAPI
    def __init__(self, num_imitation_iters, imitation_weight, config):
        self.imitation_weight = tf.get_variable("imitation_weight", initializer=float(imitation_weight),
                                                trainable=False, dtype=tf.float32)
        self.policy_weight = tf.get_variable("policy_weight", initializer=0.0, trainable=False,
                                             dtype=tf.float32)
        self.final_imitation_weight = config["model"]["custom_options"]["final_imitation_weight"]
        self.start_kl_val = config["kl_coeff"]
        self.num_imitation_iters = num_imitation_iters
        self.curr_iter = 0

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(ImitationLearningRateSchedule, self).on_global_var_update(global_vars)

        if self.curr_iter >= self.num_imitation_iters:
            self.imitation_weight.load(self.final_imitation_weight, session=self._sess)
            self.policy_weight.load(1.0, session=self._sess)
        self.curr_iter += 1


def update_kl(trainer, fetches):
    if "kl" in fetches:
        # single-agent
        trainer.workers.local_worker().for_policy(
            lambda pi: pi.update_kl(fetches["kl"]))
    else:

        def update(pi, pi_id):
            if pi_id in fetches and trainer._iteration > \
                    trainer.config['model']['custom_options']['num_imitation_iters']:
                print("Updating KL")
                pi.update_kl(fetches[pi_id]["kl"])
            else:
                if pi_id not in fetches:
                    print("No data for {}, not updating kl".format(pi_id))
                elif trainer._iteration > trainer.config['model']['custom_options']['num_imitation_iters']:
                    print("Still imitating, not updating KL yet")

        # multi-agent
        trainer.workers.local_worker().foreach_trainable_policy(update)


def loss_stats(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
    stats.update({'imitation_logprob': -policy.imitation_loss,
                  'policy_weight': policy.policy_weight,
                  'imitation_weight': policy.imitation_weight,
                  'imitation_loss': policy.imitation_weight * policy.imitation_loss})
    return stats


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ImitationLearningRateSchedule.__init__(policy, config["model"]["custom_options"]["num_imitation_iters"],
                                           config["model"]["custom_options"]["imitation_weight"], config)


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
    }


ImitationPolicy = PPOTFPolicy.with_updates(
    name="ImitationPolicy",
    before_loss_init=setup_mixins,
    stats_fn=loss_stats,
    postprocess_fn=postprocess_ppo_gae,
    grad_stats_fn=grad_stats,
    loss_fn=new_ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, ImitationLearningRateSchedule
    ])

ImitationTrainer = PPOTrainer.with_updates(name="ImitationPPOTrainer", default_policy=ImitationPolicy,
                                           after_optimizer_step=update_kl)
