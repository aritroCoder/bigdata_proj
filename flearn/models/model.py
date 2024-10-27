import torch.nn as nn

INPUT_SIZE = {
    "nab": 1
}

OUTPUT_SIZE = {
    "nab": 1
}

HIDDEN_SIZE = {
    "nab": 64
}

NUM_LAYERS = {
    "nab": 1
}

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out


def get_model_by_name(dataset, device, model):
    MODEL_DICT = {
        "grunet": GRUModel(INPUT_SIZE[dataset], HIDDEN_SIZE[dataset], OUTPUT_SIZE[dataset], NUM_LAYERS[dataset])
    }
    c_model = MODEL_DICT[model].to(device)
    return c_model
