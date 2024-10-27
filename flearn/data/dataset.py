import os
from typing import Callable, Optional
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.preprocessing import MinMaxScaler
import torch

nab_domain_to_folder = {
    "ana": "artificialNoAnomaly", 
    "awa": "artificialWithAnomaly", 
    "rae": "realAdExchange", 
    "rac": "realAWSCloudwatch", 
    "rkc": "realKnownCause", 
    "rtr": "realTraffic", 
    "rtw": "realTweets"
}

class NABData:
    dndl_links = {
        "data": {
            "link": "https://www.kaggle.com/api/v1/datasets/download/boltzmannbrain/nab",
            "path": "nab"
        }
    }

    def __init__(
            self, 
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            domain: str = "ana",
            split: float = 0.8,
            train: bool = True
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.datasets = []
        self.train = train
        self.split = split
        self.domain = domain
        self.scalar = MinMaxScaler()

        if download:
            self.download()

        self._load_data()

    def download(self):
        if os.path.exists(self.root):
            print("Dataset already downloaded.")
        else:
            print("Dataset not found. Downloading...")
            os.makedirs(self.root, exist_ok=True)
            download_and_extract_archive(self.dndl_links["data"]["link"], self.root, filename='nab.zip')

    def _load_data(self):
        folder = nab_domain_to_folder[self.domain]
        for file in os.listdir(os.path.join(self.root, folder, folder)):
            if file.endswith(".csv"):
                df = pd.read_csv(
                    os.path.join(self.root, folder, folder, file),
                    parse_dates=["timestamp"],
                )
                df = df.sort_values(by="timestamp")
                data = df["value"].values
                data = self.scalar.fit_transform(data.reshape(-1, 1)).flatten()
                if self.train:
                    data = data[:int(len(data) * self.split)]
                else:
                    data = data[int(len(data) * self.split):]
                self.datasets.append(data)
                print(f"Loaded {file} with {len(data)} samples.")

class NABDataset(Dataset):
    def __init__(self, data, seq_length, scaler, transform=None, target_transform=None):
        # self.data = torch.tensor(data, dtype=torch.float32)
        self.data=data.to(torch.float32)
        self.seq_length = seq_length
        self.transform = transform
        self.scaler = scaler
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        data = self.data[index : index + self.seq_length]
        target = self.data[index + self.seq_length]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        return data.unsqueeze(1), target

    def scale_inverse(self, data):
        return self.scaler.inverse_transform(data)


# test = NABData(root="dataset/nab", download=True)
