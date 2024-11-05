import argparse
import torch
from flearn.trainers.fedavg import Server as FedAvgServer
from flearn.trainers.customtrainer import Server as CustomServer

SERVER = {
    "fedavg": FedAvgServer,
    "custom": CustomServer
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="nab",
        help="dataset to use, default: nab",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        help="Trainer algorithm to use",
        choices=SERVER.keys()
    )
    parser.add_argument(
        "--type",
        type=str,
        help="type of dataset to use like rtw_s_5_r_8, see your dataset folder for available types",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grunet",
        help="model to use, default: grunet",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="number of rounds to run, default: 10",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    server = SERVER[args.trainer](
        dataset=args.dataset,
        pkl_folder=args.type,
        model=args.model,
        rounds=args.rounds,
        epochs=10,
        batch_size=32,
        device=device,
    )

    server.train()

main()