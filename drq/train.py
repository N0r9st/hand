import subprocess
import os
import random
import glob 
os.environ['MUJOCO_GL']="egl"

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
import wandb


from drq.learner import DrQLearner
from drq.buffer import ReplayBuffer
from drq.evaluation import evaluate
from drq.dm_env_wrapper import make_env


# NOTE: set LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

FLAGS = flags.FLAGS

# flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_string('env_name', 'reach-duplo', 'Environment name.')
flags.DEFINE_string('save_dir', './savings/', 'Dir with whatever is saved during run')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 500, 'Logging interval.') # /=5
flags.DEFINE_integer('eval_interval', 2000, 'Eval interval.') # /=5
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e5), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', True, 'Save videos during evaluation.')
flags.DEFINE_boolean('use_wandb', True, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    'drq/drq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2,
    "reach-duplo": 2,
}


def main(_):
    intermediate_vids_folder = "vids"
    if FLAGS.use_wandb:
        wandb.init(project='drq_test', config=FLAGS)
        intermediate_vids_folder += '-' + wandb.run.name
    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, intermediate_vids_folder, 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, intermediate_vids_folder, 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    kwargs = dict(FLAGS.config)
    gray_scale = kwargs.pop('gray_scale')
    image_size = kwargs.pop('image_size')

    def make_pixel_env(seed, video_folder, video_every):
        return make_env(FLAGS.env_name,
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=gray_scale,
                        video_every=video_every)

    env = make_pixel_env(FLAGS.seed, video_train_folder, 10)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder, FLAGS.eval_episodes)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    agent = DrQLearner(FLAGS.seed,
                       env.observation_space.sample()[np.newaxis],
                       env.action_space.sample()[np.newaxis], **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
        or FLAGS.max_steps // action_repeat)

    eval_returns = []
    observation, done = env.reset(), False

    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            if FLAGS.use_wandb:
                wandb.log({f'training/{k}': v for k, v in info['episode'].items()}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                if FLAGS.use_wandb:
                    wandb.log({f'training/{k}': v for k, v in update_info.items()}, step=i)


        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)
            if FLAGS.use_wandb:
                wandb.log({f'evaluation/{k}': v for k, v in eval_stats.items()}, step=i)
                vid_paths = glob.glob(os.path.join(video_eval_folder, "*.mp4"))
                vid_paths = sorted(
                    vid_paths, key=lambda x: int(os.path.basename(x).split(".")[0])
                )
                if vid_paths: wandb.log({"eval_video": wandb.Video(vid_paths[-1], fps=24, format="mp4")})

                vid_paths = glob.glob(os.path.join(video_train_folder, "*.mp4"))
                vid_paths = sorted(
                    vid_paths, key=lambda x: int(os.path.basename(x).split(".")[0])
                )
                if vid_paths: wandb.log({"train_video": wandb.Video(vid_paths[-1], fps=24, format="mp4")})

                subprocess.run(["rm", os.path.join(video_eval_folder, "*.mp4")])
                subprocess.run(["rm", os.path.join(video_train_folder, "*.mp4")])

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
