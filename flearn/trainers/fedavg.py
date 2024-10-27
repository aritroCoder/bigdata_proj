import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from tqdm import trange
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from ..models.model import get_model_by_name
from ..client.client import Client

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR")

GLR = {"grunet": {"nab": 0.001}}  # global learning rate


class Server:
    def __init__(
        self,
        dataset,
        pkl_folder,
        model,
        rounds=10,
        epochs=50,
        batch_size=32,
        device="cpu",
    ):
        self.dataset = dataset
        self.pkl_folder = pkl_folder
        self.model = model
        self.num_clients = len(os.listdir(f"{DATASET_DIR}/{dataset}/{pkl_folder}")) - 1 # -1 for test.pkl
        self.test_pkl = f"{DATASET_DIR}/{dataset}/{pkl_folder}/test.pkl"
        self.rounds = rounds
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.global_model = self.get_global_model()
        self.global_params = self.get_params_t()
        self.clients = self.setup_clients()
        print("Using federated averaging to train.")

    def get_global_model(self):
        return get_model_by_name(self.dataset, self.device, self.model)

    def get_params_t(self):
        """get model parameters"""
        with torch.no_grad():
            return [param.clone().detach() for param in self.global_model.parameters()]

    def set_params(self, model_params):
        if model_params is not None:
            with torch.no_grad():
                for param, value in zip(self.global_model.parameters(), model_params):
                    # print(type(value))
                    if isinstance(value, np.ndarray):
                        param.copy_(torch.from_numpy(value))
                    elif isinstance(value, torch.Tensor):
                        param.copy_(value)
                    else:
                        self.global_model.load_state_dict(model_params)
                        break

    def setup_clients(self):
        clients = []
        for i in range(self.num_clients):
            client = Client(
                self.dataset,
                self.pkl_folder,
                i,
                self.model,
                self.epochs,
                self.batch_size,
                GLR[self.model][self.dataset],
                self.device,
            )
            clients.append(client)
        return clients

    def train(self):
        for i in trange(self.rounds, desc="Round"):
            tqdm.write(f"\nRound {i+1}/{self.rounds}")
            csolns = []
            for c in self.clients:
                c.set_params(self.global_params)
                soln, samples = c.solve_inner()
                csolns.append((samples, soln))
            self.global_params = self.aggregate(csolns)

        self.set_params(self.global_params)
        predictions, actuals = self.test()
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.title(f"Model: {self.model}")
        plt.savefig(f"{self.dataset}_{self.pkl_folder}.png")

    def test(self):
        with open(self.test_pkl, "rb") as file:
            data = pickle.load(file)
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        self.global_model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.global_model(x_batch)
                predictions.append(output.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())

        predictions = data.scaler.inverse_transform(np.concatenate(predictions).reshape(-1, 1))
        actuals = data.scaler.inverse_transform(np.concatenate(actuals).reshape(-1, 1))
        return predictions, actuals

    def aggregate(self, wsolns):  # Weighted average
        total_weight = 0.0
        # print(f'\nwsolns:-> {wsolns}')
        base = [0] * len(wsolns[0][1])
        for w, soln in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.to(torch.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln
