import argparse
from torch.nn import functional as F


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register all hyperparameters shared between train.py and server_trainer.py."""

    # Architecture
    parser.add_argument('--embedding-size', type=int, default=1024)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--belief-size', type=int, default=200)
    parser.add_argument('--state-size', type=int, default=30)
    parser.add_argument('--action_size', default=2)
    parser.add_argument('--cnn-act', type=str, default='relu', choices=dir(F))
    parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F))

    # World model
    parser.add_argument('--free-nats', type=float, default=1.0)
    parser.add_argument('--bit-depth', type=int, default=8)
    parser.add_argument('--reward_scale', type=int, default=10)
    parser.add_argument('--pcont', action='store_true')
    parser.add_argument('--pcont_scale', type=int, default=10)
    parser.add_argument('--symbolic', action='store_true')

    # Training
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed-episodes', type=int, default=5)
    parser.add_argument('--collect-interval', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--experience-size', type=int, default=1000000)
    parser.add_argument('--world_lr', type=float, default=6e-4)
    parser.add_argument('--actor_lr', type=float, default=8e-5)
    parser.add_argument('--value_lr', type=float, default=8e-5)
    parser.add_argument('--adam-epsilon', type=float, default=1e-7)
    parser.add_argument('--grad-clip-norm', type=float, default=100.0)
    parser.add_argument('--learning-rate-schedule', type=int, default=0)

    # Policy
    parser.add_argument('--planning-horizon', type=int, default=15)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--disclam', type=float, default=0.95)
    parser.add_argument('--polyak', type=float, default=0.005,
                        help='Soft target update rate: θ_target = (1-polyak)*θ_target + polyak*θ_online')
    parser.add_argument('--expl_amount', type=float, default=0.15)
    parser.add_argument('--with_logprob', action='store_true')
    parser.add_argument('--auto_temp', action='store_true')
    parser.add_argument('--temp', type=float, default=0.003)
    parser.add_argument('--kl_balance', action=argparse.BooleanOptionalAction, default=True,
                        help='KL balancing: separate dynamics/representation loss (Dreamer v2)')
    parser.add_argument('--symlog_rewards', action=argparse.BooleanOptionalAction, default=True,
                        help='Symlog-compress rewards before reward model training (Dreamer v3)')
    parser.add_argument('--return_norm', action=argparse.BooleanOptionalAction, default=True,
                        help='Normalise returns by running 5th/95th percentile range (Dreamer v3)')

    # DonkeyCar action constraints
    parser.add_argument('--fix_speed', action='store_true', default=True)
    parser.add_argument('--throttle_base', type=float, default=0.3)
    parser.add_argument('--throttle_min', type=float, default=0.1)
    parser.add_argument('--throttle_max', type=float, default=0.5)
    parser.add_argument('--angle_min', type=float, default=-1.0)
    parser.add_argument('--angle_max', type=float, default=1.0)

    # Observation
    parser.add_argument('--grayscale', action=argparse.BooleanOptionalAction, default=False,
                        help='Use 1-channel grayscale input. Default: 3-channel RGB.')
    parser.add_argument('--observation_size', default=None)

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Apply sim-to-real augmentations during world-model training.')

    # Reproducibility & device
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Mixed-precision training: float16 on CUDA, bfloat16 on MPS.')

    # Checkpoints / resume
    parser.add_argument('--models', type=str, default='')
    parser.add_argument('--experience-replay', type=str, default='')
