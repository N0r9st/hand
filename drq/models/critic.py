from typing import Callable, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from drq.common import Batch, InfoDict, Params, PRNGKey
from drq.models.base import MLP, Encoder, TrainState
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)

class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations)(states, actions)
        return qs

class DrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DoubleCritic(self.hidden_dims)(x, actions)

def get_target(next_observations, rewards, masks, key, actor, target_critic, temp, discount, backup_entropy):
    dist = actor(next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = rewards + discount * masks * next_q

    if backup_entropy:
        target_q -= discount * masks * temp() * next_log_probs

    return target_q


def get_target_from_n_augments(batch: Batch, key, actor, target_critic, temp, discount, backup_entropy):
    keys = jax.random.split(key, batch.next_observations.shape[0])
    return jax.vmap(get_target, in_axes=(0, None, None, 0, None, None, None, None, None))(
        batch.next_observations, batch.rewards, batch.masks, keys, actor, target_critic, temp, discount, backup_entropy
    ).mean(0)



def update_critic(key: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
           temp: TrainState, batch: Batch, discount: float,
           backup_entropy: bool) -> Tuple[TrainState, InfoDict]:

    target_q = get_target_from_n_augments(batch, key, actor, target_critic, temp, discount, backup_entropy)
    m_repeats, batch_size, *dims = batch.observations.shape
    target_q = jnp.repeat(target_q, repeats=m_repeats, axis=0)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations.reshape((m_repeats*batch_size, *dims)),
                                        jnp.repeat(batch.actions, repeats=m_repeats, axis=0))
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
