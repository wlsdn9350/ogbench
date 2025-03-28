import copy
from typing import Any

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize
from utils.q_mixer_networks import QFunctionMixerCore


# The QMixer agent is defined as a PyTreeNode so it can easily be updated/replaced.
class QMixerAgent(flax.struct.PyTreeNode):
    """QMixer agent"""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        observations = batch['observations']  # shape: (B, V, D)
        actions = batch['actions']    # ground truth or teacher-forcing tokens
        goals = batch['actor_goals']
        
        if len(observations.shape) == 2:
            observations = jnp.expand_dims(observations, axis=1)
            actions = jnp.expand_dims(actions, axis=1)
            goals = jnp.expand_dims(goals, axis=1)
        
        actions = continuous_to_discrete(actions, self.config['action_max'], self.config['action_min'], self.config['num_bins'])
        
        rewards = batch['rewards']
        rewards = jnp.expand_dims(rewards, axis=-1)

        current_dist, _ = self.network.select('q_predictors')(observations, goals, action_seq=actions, params=grad_params)

        current_q, _ = self.network.select('q_predictors')(observations, goals, action_seq=None)
        next_q, _ = self.network.select('target_q_predictors')(observations, goals, action_seq=None)

        current_q_max = current_q.max(axis=-1)
        next_q_max = next_q.max(axis=-1)

        td_targets = jnp.zeros_like(next_q_max)

        # All dimensions except last use next action dim
        # print(td_targets.shape, rewards.shape, next_q_max[..., 0].shape)
        td_targets = td_targets.at[..., :-1].set(current_q_max[..., 1:])
        td_targets = td_targets.at[..., -1].set(rewards + self.config['discount'] * next_q_max[..., 0])

        #TODO: Implement mc_returns

        action_mask = jax.nn.one_hot(actions, num_classes=self.config['num_bins']).astype(jnp.float32)

        current_q = (current_dist * action_mask).sum(axis=-1)
        td_error = 0.5 * jnp.mean((current_q - td_targets) ** 2)

        # Conservative regularization
        B, V, A, N = current_dist.shape
        non_action_mask = 1 - action_mask
        non_action_q = (current_dist ** 2 * non_action_mask).sum(axis=-1)
        conservative_loss = (self.config['alpha'] / (2 * (N - 1))) * non_action_q.mean()

        info["td_error"] = td_error
        info["conservative_loss"] = conservative_loss
        info["total_loss"] = td_error + conservative_loss

        loss = td_error + conservative_loss
        
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with accompanying info."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        
        # Here we assume that network.apply_loss_fn is available to update parameters.
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'q_predictors')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor network."""
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
            observations = jnp.expand_dims(observations, axis=1)
            goals = jnp.expand_dims(goals, axis=0)
            goals = jnp.expand_dims(goals, axis=1)

        _, predicted_actions = self.network.select('target_q_predictors')(observations, goals, action_seq=None)
        actions = discrete_to_continuous(predicted_actions, self.config['action_max'],
                                         self.config['action_min'], self.config['num_bins'])  # Placeholder: directly use predicted discrete actions.
        
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new QMixer agent.
        
        Args:
            seed: Random seed.
            ex_observations: Example observation batch.
            ex_actions: Example action batch (for discrete actions, expect max action value).
            config: Configuration dictionary.
        """
        print("Creating QMixer agent")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if len(ex_observations.shape) == 2:
            ex_observations = jnp.expand_dims(ex_observations, axis=1)
            ex_actions = jnp.expand_dims(ex_actions, axis=1)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = jnp.max(ex_actions) + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoder.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
        
        if config['discrete']:
            q_predictors = None
            raise NotImplementedError("Discrete actions not supported yet.")
        
        else:
            q_predictors = QFunctionMixerCore(
                num_tokens=ex_observations.shape[1],
                state_dim=config['feature_dim'],
                num_action_dims=action_dim,
                num_bins=config.get('num_bins', 256),
                joint_embed_dim=config.get('joint_embed_dim', 256),
                num_mixer_blocks=config.get('num_mixer_blocks', 1),
                mixer_token_hidden=config.get('mixer_token_hidden', 256),
                mixer_channel_hidden=config.get('mixer_channel_hidden', 256),
                gc_encoder=encoders.get('actor')
            )

        ex_discrete_actions = continuous_to_discrete(ex_actions, config['action_max'], config['action_min'], config['num_bins'])

        network_info = dict(
            q_predictors=(q_predictors, (ex_observations, ex_goals, ex_discrete_actions)),
            target_q_predictors=(copy.deepcopy(q_predictors), (ex_observations, ex_goals, ex_discrete_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def discretize(actions):
    actions = jnp.array(actions)
    action_max = jnp.max(actions, axis=0)
    action_min = jnp.min(actions, axis=0)

    return action_max, action_min

def discrete_to_continuous(action_logits, action_max, action_min, num_bins):
    # Converts tensor of shape (B, V, A, N) to (B, V, A)
    discrete_actions = action_logits.astype(jnp.float32)
    continuous_actions = (discrete_actions / (num_bins - 1)) * (
        action_max - action_min
    ) + action_min
    return continuous_actions

def continuous_to_discrete(actions, action_max, action_min, num_bins):
    actions = (actions - action_min) / (action_max - action_min)
    actions = actions * (num_bins - 1)
    actions = jnp.round(actions).astype(jnp.int32)  # shape (B, V, A)
    actions = jnp.clip(actions, 0, num_bins - 1)
    # actions = jax.nn.one_hot(actions, num_classes=num_bins)  # shape (B, V, A, num_bins)
    return actions

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qmixer',  # Agent name.
            lr=3e-4,              # Learning rate.
            batch_size=64,      # Batch size.
            discount=0.99,
            alpha=1.0,
            tau=0.005,  # Target network update rate.
            const_std=True,
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            dataset_class='GCDataset',
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),
            feature_dim=256,
            action_max=1.0,
            action_min=-1.0,
            num_bins=256,
        )
    )
    return config
