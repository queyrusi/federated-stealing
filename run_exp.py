import argparse
from steal_cgan import ClueStealing
from models.cgan import Generator


def get_steal_args():
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--lr", type=float, default=0.0032)
    args.add_argument("--b1", type=float, default=0.1)  # default 0.5
    args.add_argument("--b2", type=float, default=0.1)  # default 0.999

    args.add_argument("--nc", type=int, default=1)
    args.add_argument("--ngf", type=int, default=8)

    args.add_argument("--nz", type=int, default=100)
    args.add_argument("--defense", type=str, default="")
    args.add_argument("--n_epoch", type=int, default=400)
    args.add_argument("--n_batch", type=int, default=10)
    args.add_argument("--img_size", type=int, default=32)
    args.add_argument("--n_thieves", type=int, default=10)
    args.add_argument("--num_QperA", type=int, default=500)
    args.add_argument("--n_samples4fid", type=int, default=2048)
    args.add_argument("--dataset", default="FashionMNIST")
    args.add_argument("--counter_measure", choices=["JPEG"], default="")
    args.add_argument("--n_labels", type=int, default=2)

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = get_steal_args()
    if args.dataset in ("MNIST", "FashionMNIST"):
        args.img_size = 32
        args.n_labels = 10
    elif args.dataset == "CelebA":
        args.img_size = 64
        args.defense = "None"
        args.n_labels = 2
        args.ngf = 64

    cs = ClueStealing(args)
    cs.steal()
