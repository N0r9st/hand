import glob
import os
import random
import subprocess

os.environ['MUJOCO_GL']="egl"

import numpy as np
import tqdm
import wandb

from drq.buffer import ReplayBuffer
from drq.env import make_env
from drq.evaluation import evaluate
from drq.learner import DrQLearner

# NOTE: set LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2,
    "reach-duplo": 2,
}


def main(args):
    intermediate_vids_folder = "vids"
    
    video_train_folder = os.path.join(args['save_dir'], intermediate_vids_folder, 'train')
    video_eval_folder = os.path.join(args['save_dir'], intermediate_vids_folder, 'eval')

    if args["action_repeat"] is not None:
        action_repeat = args['action_repeat']
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(args["env_name"], 2)

    gray_scale = args["gray_scale"] # kwargs.pop('gray_scale')
    image_size = args['image_size']

    def make_pixel_env(seed, video_folder, video_every):
        return make_env(args["env_name"],
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=gray_scale,
                        video_every=video_every)

    env = make_pixel_env(args["seed"], video_train_folder, 10)
    eval_env = make_pixel_env(args["seed"] + 42, video_eval_folder, args["eval_episodes"])

    np.random.seed(args["seed"])
    random.seed(args["seed"])

    replay_buffer_size = args['replay_buffer_size']
    agent = DrQLearner(args["seed"],
                       env.observation_space.sample()[np.newaxis],
                       env.action_space.sample()[np.newaxis], **args['learner_config'])

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
        or args["max_steps"] // action_repeat)

    eval_returns = []
    observation, done = env.reset(), False

    if args['wandb_project'] is not None:
        wandb.init(project=args['wandb_project'], config=args)
        intermediate_vids_folder += '-' + wandb.run.name

    for i in tqdm.tqdm(range(1, args["max_steps"] // action_repeat + 1),
                       smoothing=0.1,
                       disable=not args["tqdm"]):
        if i < args["start_training"]:
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
            if args['wandb_project'] is not None:
                wandb.log({f'training/{k}': v for k, v in info['episode'].items()}, step=i)

        if i >= args["start_training"]:
            batch = replay_buffer.sample(args["batch_size"])
            update_info = agent.update(batch)

            if i % args["log_interval"] == 0:
                if args['wandb_project'] is not None:
                    wandb.log({f'training/{k}': v for k, v in update_info.items()}, step=i)


        if i % args["eval_interval"] == 0:
            eval_stats = evaluate(agent, eval_env, args["eval_episodes"])
            if args['wandb_project'] is not None:
                wandb.log({f'evaluation/{k}': v for k, v in eval_stats.items()}, step=i)
                if args["save_video"]:
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
            np.savetxt(os.path.join(args["save_dir"], f'{args["seed"]}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    from drq.args import args
    main(args)
