""" Define parameters for skill chaining algorithms. """

import argparse

from robot_learning.config import (
    add_method_arguments,
    add_il_arguments,
    add_gail_arguments,
    add_ppo_arguments,
    str2bool,
    str2intlist,
    str2list,
)


def add_ps_arguments(parser):
    parser.add_argument("--ps_ckpts", type=str2list, default=None)
    parser.add_argument("--ps_demo_paths", type=str2list, default=None)

    # T-STAR
    parser.add_argument(
        "--ps_use_tstar",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--ps_tstar_reward", type=float, default=10000.0)
    parser.add_argument(
        "--ps_tstar_reward_type",
        type=str,
        default="amp",
        choices=["vanilla", "gan", "d", "amp"],
    )

    # ps environment initialization
    parser.add_argument(
        "--ps_use_terminal_states",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--ps_env_init_from_dist",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ps_env_init_from_states",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--ps_load_init_states",
        type=str2list,
        default=None,
        help="path to pickle file of initial states",
    )

    # ps training
    parser.add_argument(
        "--ps_epochs",
        type=int,
        default=500,
        help="number of training epochs of policy sequencing training",
    )
    parser.add_argument(
        "--ps_sub_policy_update_steps",
        type=int,
        default=100000,
        help="training steps for each sub-policy update per each epoch",
    )
    parser.add_argument("--ps_entropy_loss_coeff", type=float, default=0.0)
    parser.add_argument(
        "--ps_discriminator_loss_type",
        type=str,
        default="lsgan",
        choices=["gan", "lsgan"],
    )
    parser.add_argument("--ps_discriminator_lr", type=float, default=1e-4)
    parser.add_argument(
        "--ps_discriminator_mlp_dim", type=str2intlist, default=[256, 256]
    )
    parser.add_argument(
        "--ps_discriminator_activation",
        type=str,
        default="tanh",
        choices=["relu", "elu", "tanh"],
    )
    parser.add_argument("--ps_discriminator_update_freq", type=int, default=4)
    parser.add_argument(
        "--ps_discriminator_replay_buffer", type=str2bool, default=False
    )
    parser.add_argument("--ps_discriminator_buffer_size", type=int, default=100)
    parser.add_argument("--ps_grad_penalty_coeff", type=float, default=10.0)
    parser.add_argument(
        "--ps_rl_algo", type=str, default="gail", choices=["ppo", "sac", "td3", "gail"]
    )


def create_skill_chaining_parser():
    """
    Creates the argparser for skill chaining.
    """
    parser = argparse.ArgumentParser(
        "Skill Chaining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )

    parser.add_argument("--seed", type=int, default=123)

    # environment
    parser.add_argument(
        "--env",
        type=str,
        default="IKEASawyerDense-v0",
        help="environment name",
    )

    # algorithm
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo", "ddpg", "td3", "bc", "gail", "dac", "ps"],
    )

    # method
    add_method_arguments(parser)

    args, unparsed = parser.parse_known_args()

    if args.algo == "ps":
        add_il_arguments(parser)
        add_gail_arguments(parser)
        add_ppo_arguments(parser)
        add_ps_arguments(parser)
        parser.set_defaults(max_ob_norm_step=0)

    parser.set_defaults(log_root_dir="log")
    parser.set_defaults(wandb=False)
    parser.set_defaults(wandb_project=None)
    parser.set_defaults(policy_mlp_dim=[128, 128])
    parser.set_defaults(max_global_step=int(1e8))

    # PPO/GAIL
    parser.set_defaults(reward_scale=0.05)
    parser.set_defaults(rollout_length=1024)
    parser.set_defaults(evaluate_interval=50)
    parser.set_defaults(ckpt_interval=50)

    # GAIL
    parser.set_defaults(gail_reward="amp")
    parser.set_defaults(gail_use_action=False)
    parser.set_defaults(gail_env_reward=0.5)
    parser.set_defaults(discriminator_loss_type="lsgan")
    parser.set_defaults(demo_low_level=True)

    # furniture
    if "furniture" in args.env.lower() or "IKEA" in args.env:
        from furniture.config.furniture import add_argument as add_furniture_arguments
        from furniture.config import add_env_specific_arguments

        add_furniture_arguments(parser)
        add_env_specific_arguments(args.env, parser)

        parser.set_defaults(reset_robot_after_attach=True)
        parser.set_defaults(phase_ob=True)

    return parser


def argparser():
    """ Directly parses the arguments. """
    parser = create_skill_chaining_parser()
    args, unparsed = parser.parse_known_args()

    return args, unparsed
