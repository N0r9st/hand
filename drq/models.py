from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn


from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x

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


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist



class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


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


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> tfd.Distribution:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x,
                                                                   temperature)
