import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from drq.common import Batch, InfoDict, PRNGKey
from drq.models import (DrQDoubleCritic, DrQPolicy, Temperature, TrainState,
                        sample_actions, update_actor, update_critic,
                        update_temperature)


def target_update(critic: TrainState, target_critic: TrainState, tau: float) -> TrainState:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)

def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)

def batched_random_crop_ntimes(key, imgs, n, padding=4):
    keys = jax.random.split(key, n)
    n_augs_imgs = jnp.repeat(imgs[jnp.newaxis, ...], repeats=n, axis=0)
    return jax.vmap(batched_random_crop, in_axes=(0, 0, None))(keys, n_augs_imgs, padding)

@functools.partial(jax.jit, static_argnames=("update_target", "num_aug", "num_aug_target"))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
    temp: TrainState, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool, num_aug: int, num_aug_target: int
) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    # observations = batched_random_crop(key, batch.observations)
    m_observations = batched_random_crop_ntimes(key, batch.observations, n=num_aug)
    
    rng, key = jax.random.split(rng)
    # next_observations = batched_random_crop(key, batch.next_observations)
    k_next_observations = batched_random_crop_ntimes(key, batch.next_observations, n=num_aug_target)

    batch = batch._replace(observations=m_observations,
                           next_observations=k_next_observations)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            backup_entropy=True)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy(
        add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class DrQLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1,
                 num_aug: int = 1,
                 num_aug_target: int = 1):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)


        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor_params = actor_def.init(*[actor_key, observations])['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                             params=actor_params,
                             tx=optax.adam(learning_rate=actor_lr))


        critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                     cnn_padding, latent_dim)
        critic_params = critic_def.init(*[critic_key, observations, actions])['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                              params=critic_params,
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic_params = critic_def.init(*[critic_key, observations, actions])['params']
        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=target_critic_params, tx=optax.adam(learning_rate=critic_lr))


        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                            params=temp_params,
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 0
        self.num_aug = num_aug
        self.num_aug_target = num_aug_target

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0, self.num_aug, self.num_aug_target)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
