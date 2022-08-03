import argparse
import ml_collections

def tuple_from_string(s):
    return tuple(int(x) for x in s.split("-"))

def get_cmd_args():
    parser = argparse.ArgumentParser()

    #  -----  train setup -------
    parser.add_argument("--env-name", type=str, default="reach-duplo")
    parser.add_argument("--save-dir", type=str, default="./savings/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=int(5e5))
    parser.add_argument("--start-training", type=int, default=int(1e3))
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--no-tqdm", action="store_true", default=False)
    parser.add_argument("--save-video", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=None)

    # ----- learner args ----
    

    parser.add_argument("--algo", type=str, default='drq')

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--temp-lr", type=float, default=3e-4)

    parser.add_argument("--hidden-dims", type=str, default="256-256")

    parser.add_argument("--cnn-features", type=str, default="32-32-32-32")
    parser.add_argument("--cnn-strides", type=str, default="2-1-1-1")
    parser.add_argument("--cnn-padding", type=str, default='VALID')
    parser.add_argument("--latent-dim", type=int, default=50)

    parser.add_argument("--discount", type=float, default=0.99)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-update-period", type=int, default=1)

    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--target-entropy", type=float, default=None)

    parser.add_argument("--replay-buffer-size", type=int, default=100_000)

    parser.add_argument("--gray-scale", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=84)

    parser.add_argument("--num-aug", type=int, default=2)
    parser.add_argument("--num-aug-target", type=int, default=2)

    return parser.parse_args()

def make_args_dict():
    cmd_args = get_cmd_args()
    args = dict(
        algo=cmd_args.algo,

        env_name=cmd_args.env_name,
        save_dir=cmd_args.save_dir,
        seed=cmd_args.seed,
        eval_episodes=cmd_args.eval_episodes,
        log_interval=cmd_args.log_interval,
        eval_interval=cmd_args.eval_interval,
        batch_size=cmd_args.batch_size,
        max_steps=cmd_args.max_steps,
        start_training=cmd_args.start_training,
        action_repeat=cmd_args.action_repeat,
        tqdm=not cmd_args.no_tqdm,
        save_video=cmd_args.save_video,
        wandb_project=cmd_args.wandb_project,

        learner_config=dict(
            actor_lr=cmd_args.actor_lr,
            critic_lr=cmd_args.critic_lr,
            temp_lr=cmd_args.temp_lr,
            hidden_dims=tuple_from_string(cmd_args.hidden_dims),
            cnn_features=tuple_from_string(cmd_args.cnn_features),
            cnn_strides=tuple_from_string(cmd_args.cnn_strides),
            cnn_padding=cmd_args.cnn_padding,
            latent_dim=cmd_args.latent_dim,
            discount=cmd_args.discount,
            tau=cmd_args.tau,
            target_update_period=cmd_args.target_update_period,
            init_temperature=cmd_args.init_temperature,
            target_entropy=cmd_args.target_entropy,
            num_aug=cmd_args.num_aug,
            num_aug_target=cmd_args.num_aug_target,
        ),

        gray_scale=cmd_args.gray_scale,
        image_size=cmd_args.image_size,
        replay_buffer_size=cmd_args.replay_buffer_size,

    )
    return args

args = make_args_dict()
