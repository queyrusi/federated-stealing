from gan_alg.train_cgan import TrainConditionalGAN
from gan_alg.steal_cgan import StealConditionalGAN
import argparse
import torch

from gan_alg.show import show_cgan_result


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--z_dim', type=int, default=128)
    args.add_argument('--lr', type=float, default=0.0002)

    args.add_argument('--lr_g', type=float, default=0.0002, help='for Bayes')
    args.add_argument('--lr_d', type=float, default=0.0002, help='for Bayes')

    args.add_argument('--train_epoch', type=int, default=50)

    args = args.parse_args()
    return args


def get_steal_args():
    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--z_dim', type=int, default=128)
    args.add_argument('--lr', type=float, default=0.0002)

    args.add_argument('--n_epoch', type=int, default=20)
    args.add_argument('--n_batch', type=int, default=40)
    args.add_argument('--s_type', type=str, default='Prob', help='Prob or Imgs, for Bayes target')

    args.add_argument('--s_layers', type=int, default=2, help='0, 1, 2, last layer outputs for steal CGAN')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    # args = get_args()
    # t = TrainConditionalGAN(args)
    # t.train()

    args = get_steal_args()
    s = StealConditionalGAN(args)
    s.steal()


