from typing import Sequence, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class MixerBlock(nn.Module):
    """
    Mixer block for MLP-Mixer (token mixing + channel mixing).

    Attributes:
        num_tokens: Number of tokens (sequence length).
        embed_dim: Embedding dimension.
        hidden_dim_tokens: Hidden layer size for token mixing.
        hidden_dim_channels: Hidden layer size for channel mixing.
    """
    num_tokens: int
    embed_dim: int
    hidden_dim_tokens: int
    hidden_dim_channels: int

    def setup(self):
        self.token_norm = nn.LayerNorm()
        self.token_dense1 = nn.Dense(self.hidden_dim_tokens)
        self.token_dense2 = nn.Dense(self.num_tokens)
        self.channel_norm = nn.LayerNorm()
        self.channel_dense1 = nn.Dense(self.hidden_dim_channels)
        self.channel_dense2 = nn.Dense(self.embed_dim)

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: (B, num_tokens, embed_dim)
        if mask is not None:
            mask_exp = jnp.expand_dims(mask, axis=-1)
            x = x * mask_exp
        
        y = self.token_norm(x)
        # Transpose for token mixing across the token dimension.
        y = jnp.transpose(y, (0, 2, 1))
        y = self.token_dense1(y)
        y = nn.gelu(y)
        y = self.token_dense2(y)
        y = jnp.transpose(y, (0, 2, 1))

        if mask is not None:
            y = y * mask_exp
        x = x + y  # residual connection

        # Channel mixing within each token.
        z = self.channel_norm(x)
        z = self.channel_dense1(z)
        z = nn.gelu(z)
        z = self.channel_dense2(z)
        if mask is not None:
            z = z * mask_exp

        return x + z  # final residual connection

class QFunctionMixerCore(nn.Module):
    """
    Q-function module with integrated MLP-Mixer.
    
    This module predicts Q-value distributions for each action dimension by combining
    state embeddings and previous action tokens through Mixer blocks in an autoregressive manner.
    
    Attributes:
        num_tokens: Number of state tokens (sequence length).
        state_dim: Dimension of state features.
        num_action_dims: Total number of action dimensions.
        num_bins: Number of discrete bins for each action.
        joint_embed_dim: Joint embedding dimension for mapping actions.
        num_mixer_blocks: Number of MixerBlock layers.
        mixer_token_hidden: Hidden dimension for token mixing.
        mixer_channel_hidden: Hidden dimension for channel mixing.
    """
    num_tokens: int
    state_dim: int
    num_action_dims: int
    num_bins: int = 256
    joint_embed_dim: int = 128
    num_mixer_blocks: int = 2
    mixer_token_hidden: int = 64
    mixer_channel_hidden: int = 64
    gc_encoder: nn.Module = None

    def setup(self):
        self.action_embed = nn.Embed(num_embeddings=self.num_bins,
                                     features=self.joint_embed_dim,
                                     name="action_embed")
        # Parameter for previous token embeddings:
        self.prev_tokens = self.param("prev_tokens",
                                      nn.initializers.normal(stddev=0.1),
                                      (1, 1, 1, self.joint_embed_dim))

        self.mixer_block = MixerBlock(num_tokens=self.num_tokens * (self.num_action_dims + 2),
                       embed_dim=self.state_dim,
                       hidden_dim_tokens=self.mixer_token_hidden,
                       hidden_dim_channels=self.mixer_channel_hidden)
        self.output_head = nn.Dense(self.num_bins, name="output_head")

    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray, action_seq: Optional[jnp.ndarray] = None):
        """
        Args:
            observations: Array of shape (B, V, state_dim) containing input observations.
            action_seq: (Optional) Array of shape (B, V, num_action_dims) with ground truth discrete
                        action tokens for teacher forcing; if None the module operates in evaluation mode.
                        
        Returns:
            A tuple (Q_values, predicted_actions) where:
              - Q_values is a (B, V, num_action_dims, num_bins) array representing predicted Q-value distributions.
              - predicted_actions is a (B, V, num_action_dims) array of predicted discrete action tokens.
        """
        B, V, _ = observations.shape

        if self.gc_encoder is not None:
            observations, goals = self.gc_encoder(observations, goals, concat_encoded=False)
        else:
            raise ValueError("GC encoder must be provided.")
        
        features = jnp.concatenate([observations, goals], axis=1)

        assert V == self.num_tokens, "Input sequence length must match num_tokens."
        Q_values_list = []
        predicted_actions_list = []  # only used if action_seq is None
        # Repeat the prev_tokens across batch, tokens, and action dimensions.
        prev_embed_tokens = jnp.tile(self.prev_tokens, (B, V, self.num_action_dims, 1))

        for dim in range(self.num_action_dims):
            state_embed = features  # (B, V * 2, state_dim)
            if dim == 0:
                prev_embeds = prev_embed_tokens
            else:
                if action_seq is not None:
                    # Use ground truth tokens for teacher forcing.
                    new_prev_tokens = action_seq[:, :, :dim]
                    
                else:
                    # Use previously predicted tokens.
                    new_prev_tokens = jnp.stack(predicted_actions_list, axis=2)  # (B, V, dim)
                # Embed new tokens: (B, V, dim, joint_embed_dim)
                prev_embeds_new = self.action_embed(new_prev_tokens)
                # Concatenate with remaining pre-learned embeddings.
                prev_embeds = jnp.concatenate([prev_embeds_new, prev_embed_tokens[:, :, dim:, :]], axis=2)
            
            # Reshape to merge action dimensions: (B, -1, joint_embed_dim)
            prev_embeds = jnp.reshape(prev_embeds, (B, -1, self.joint_embed_dim))
            # Joint input: concatenate state embeddings with action embeddings along the token axis.
            joint_input = jnp.concatenate([state_embed, prev_embeds], axis=1)
            # Determine the target token range for current prediction.
            target_dim = state_embed.shape[1] + dim + 1
            x = joint_input
            total_tokens = x.shape[1]
            
            # Create a binary mask: ones for tokens to be updated, zeros for the rest.
            mask = jnp.concatenate([
                jnp.ones((B, target_dim)),
                jnp.zeros((B, total_tokens - target_dim))
            ], axis=1)
            
            # Apply all Mixer blocks.
            for _ in range(self.num_mixer_blocks):
                x = self.mixer_block(x, mask=mask)
            
            # Final output head to obtain Q-value logits per bin.
            q_logits = self.output_head(x)
            q_logits = jax.nn.sigmoid(q_logits)
            # Select the token slice corresponding to the current action dimension.
            q_logits = q_logits[:, (target_dim - V): target_dim, :]  # (B, V, num_bins)
            Q_values_list.append(jnp.expand_dims(q_logits, axis=2))
            
            # In evaluation mode (no teacher forcing), choose the argmax bin.
            if action_seq is None:
                pred_bins = jnp.argmax(q_logits, axis=-1)  # (B, V)
                predicted_actions_list.append(pred_bins)
        Q_values = jnp.concatenate(Q_values_list, axis=2)  # (B, V, num_action_dims, num_bins)
        
        if action_seq is None:
            predicted_actions = jnp.stack(predicted_actions_list, axis=2)  # (B, V, num_action_dims)
        else:
            predicted_actions = action_seq
        
        return Q_values, predicted_actions
