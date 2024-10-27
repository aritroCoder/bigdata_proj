import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from ..models.model import get_model_by_name

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR")

class Client:
    def __init__(
        self,
        dataset,
        pkl_folder,
        client_idx,
        model,
        epochs=50,
        batch_size=32,
        lr=0.001,
        device="cpu",
    ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = get_model_by_name(dataset, device, model)
        self.pickle_file = f"{DATASET_DIR}/{dataset}/{pkl_folder}/{client_idx}.pkl"
        with open(self.pickle_file, "rb") as file:
            data = pickle.load(file)
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_samples = len(data)

    def get_params_t(self):
        '''get model parameters'''
        with torch.no_grad():
            return [param.clone().detach() for param in self.model.parameters()]

    def set_params(self, model_params):
        if model_params is not None:
            with torch.no_grad():
                for param, value in zip(self.model.parameters(), model_params):
                    # print(type(value))
                    if isinstance(value, np.ndarray):
                        param.copy_(torch.from_numpy(value))
                    elif isinstance(value, torch.Tensor):
                        param.copy_(value)
                    else:
                        self.model.load_state_dict(model_params)
                        break

    def solve_inner(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for x_batch, y_batch in self.dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        soln = self.get_params_t()

        return soln, self.num_samples
