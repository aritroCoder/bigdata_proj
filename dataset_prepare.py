import os
import pickle
import numpy as np
import random
import torch
from argparse import Namespace
# from torchvision import transforms

from flearn.data.dataset import NABData, NABDataset
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.getenv(
    "DATASET_DIR"
)  # /home/fishnak/Documents/Coding/python/bigdata_proj/dataset

DATASET = {"nab": (NABData, NABDataset)}


def preprocess(args: Namespace) -> None:
    print(args)
    dataset_dir = f"{CURRENT_DIR}/{args.dataset}"
    pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.domain}_s_{args.seq_length}_f_{str(args.fraction).split('.')[1]}"
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    transform =None
    target_transform = None

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(pickles_dir):
        os.makedirs(pickles_dir, exist_ok=True)
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "nab":
        trainset = ori_dataset(
            root=os.path.join(dataset_dir, "raw_data"),
            transform=None,
            target_transform=None,
            download=True,
            domain=args.domain,
            split=args.fraction,
            train=True,
        )
        testset = ori_dataset(
            root=os.path.join(dataset_dir, "raw_data"),
            transform=None,
            target_transform=None,
            download=True,
            domain=args.domain,
            split=args.fraction,
            train=False,
        )
    else:  # Should not be executed
        trainset = ori_dataset(
            dataset_dir, train=True, download=True, transform=None
        )
        testset = ori_dataset(dataset_dir, train=False, transform=None)

    all_trainsets = []
    for data in trainset.datasets:
        all_trainsets.append(
            target_dataset(
                data, args.seq_length, trainset.scalar, transform, target_transform
            )
        )

    test_data = []
    for data in testset.datasets:
        data = torch.Tensor(data)
        test_data.extend(data)
    test_data = torch.stack(test_data)
    testset = target_dataset(
        test_data, args.seq_length, testset.scalar, transform, target_transform
    )

    client_id = 0
    for dataset in all_trainsets:
        with open(pickles_dir + f"/{client_id}.pkl", "wb") as f:
            pickle.dump(dataset, f)
        client_id += 1

    with open(pickles_dir + "/test.pkl", "wb") as f:
        pickle.dump(testset, f)
