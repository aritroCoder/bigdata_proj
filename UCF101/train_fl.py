import argparse
import datetime
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import ConvLSTM
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Conv-LSTM model")
    parser.add_argument("--checkpoints_dir", type=str, default="./chkpt", help="Directory for saving model checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints")
    parser.add_argument("--image_height", type=int, default=224, help="Height of input images")
    parser.add_argument("--image_width", type=int, default=224, help="Width of input images")
    parser.add_argument("--sequence_length", type=int, default=16, help="Length of input sequences")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism in LSTM")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients in federated learning")
    parser.add_argument("--rounds", type=int, default=2, help="Number of federated learning rounds")
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, save_dir):
    save_path = Path(save_dir) / f"checkpoint_{epoch:04}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def average_weights(client_weights):
    """Aggregate client weights using Federated Averaging."""
    avg_weights = client_weights[0]
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = avg_weights[key] / len(client_weights)
    return avg_weights

def average_weights_with_bias(client_weights, bias_factor=0.65):
    """
    Aggregate client weights using a biased weighted average.
    
    Args:
        client_weights (list[dict]): List of client model weights (state_dicts).
        bias_factor (float): Factor to adjust client contributions randomly.
                             0.65 applies a slight downscale, 1.35 upscales.
    
    Returns:
        dict: Aggregated global weights.
    """
    # Initialize the aggregated weights dictionary with zeros
    avg_weights = {key: torch.zeros_like(val, dtype=torch.float64) for key, val in client_weights[0].items()}
    
    total_weight = 0.0  # Total effective weight for normalization
    
    # Iterate through all client weights
    for i, client_state in enumerate(client_weights):
        # Determine the bias-adjusted weight
        random_factor = random.random()
        bias_weight = bias_factor if random_factor < 0.1 else (2 - bias_factor)
        
        # Add biased contributions to the aggregated weights
        for key in client_state.keys():
            avg_weights[key] += client_state[key].to(torch.float64) * bias_weight
        
        # Accumulate the total effective weight
        total_weight += bias_weight

    # Normalize the aggregated weights
    for key in avg_weights.keys():
        avg_weights[key] /= total_weight

    return avg_weights


def main():
    args = parse_args()

    # Logger setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training script")

    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (3, args.image_height, args.image_width)
    dataset_path = "/mnt/Data/raghav/bigdata_proj/ucf101_arkadeep/data/ucf101_frames"
    split_path = "/mnt/Data/raghav/bigdata_proj/ucf101_arkadeep/ucfTrainTestlist"

    # Global model
    init_dataset = Dataset(
                dataset_path=dataset_path,
                split_path=split_path,
                split_number=1,  # Split logic may vary
                input_shape=image_shape,
                sequence_length=args.sequence_length,
                training=True,
            )
    global_model = ConvLSTM(
        num_classes=init_dataset.num_classes,  # Adjust as per your dataset
        latent_dim=args.latent_dim,
        lstm_layers=args.lstm_layers,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        attention=args.attention,
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=2e-2)

    start_epoch = 0
    if args.resume and (checkpoint_path := Path(args.checkpoint_model)).exists():
        start_epoch = load_checkpoint(global_model, optimizer, checkpoint_path, device)

    # Federated Learning
    for round in range(args.rounds):
        logger.info(f"Starting round {round + 1}/{args.rounds}")
        client_weights = []

        # Each client trains on its local data
        for client in range(args.num_clients):
            epoch_metrics = {"loss": [], "acc": []}
            logger.info(f"Training client {client + 1}/{args.num_clients}")

            # Local dataset for the client
            train_dataset = Dataset(
                dataset_path=dataset_path,
                split_path=split_path,
                split_number=client + 1,  # Split logic may vary
                input_shape=image_shape,
                sequence_length=args.sequence_length,
                training=True,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )

            # Clone the global model to use for local training
            client_model = ConvLSTM(
                num_classes=init_dataset.num_classes,
                latent_dim=args.latent_dim,
                lstm_layers=args.lstm_layers,
                hidden_dim=args.hidden_dim,
                bidirectional=args.bidirectional,
                attention=args.attention,
            ).to(device)
            client_model.load_state_dict(global_model.state_dict())

            client_optimizer = torch.optim.Adam(client_model.parameters(), lr=2e-3)

            # Training loop
            client_model.train()
            for epoch in range(start_epoch, args.num_epochs):
                for batch_i, (X, y) in enumerate(train_dataloader):
                    if X.size(0) == 1:
                        continue

                    image_sequences = X.to(device)
                    labels = y.to(device)

                    client_optimizer.zero_grad()
                    client_model.lstm.reset_hidden_state()
                    predictions = client_model(image_sequences)
                    loss = criterion(predictions, labels)
                    acc = (predictions.argmax(1) == labels).float().mean().item()
                    loss.backward()
                    client_optimizer.step()
                    epoch_metrics["loss"].append(loss.item())
                    epoch_metrics["acc"].append(acc)
                    logger.info(
                        f"Epoch [{epoch}/{args.num_epochs}] Batch [{batch_i}/{len(train_dataloader)}] Client {client} "
                        f"Loss: {np.mean(epoch_metrics['loss']):.3f}, Acc: {np.mean(epoch_metrics['acc']):.3f}"
                    )

            # Collect client weights
            client_weights.append(client_model.state_dict())

        # Aggregate weights using Federated Averaging
        # global_weights = average_weights(client_weights)
        global_weights = average_weights(client_weights, args)
        global_model.load_state_dict(global_weights)

        # Evaluation
        global_model.eval()
        test_dataset = Dataset(
            dataset_path=dataset_path,
            split_path=split_path,
            split_number=3,
            input_shape=image_shape,
            sequence_length=args.sequence_length,
            training=False,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = X.to(device)
            labels = y.to(device)

            with torch.no_grad():
                global_model.lstm.reset_hidden_state()
                predictions = global_model(image_sequences)

            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean().item()
            test_metrics["loss"].append(loss.item())
            test_metrics["acc"].append(acc)

        logger.info(
            f"Round [{round + 1}/{args.rounds}] Test Loss: {np.mean(test_metrics['loss']):.3f}, "
            f"Test Accuracy: {np.mean(test_metrics['acc']):.3f}"
        )

        # Save checkpoint
        if (round + 1) % args.checkpoint_interval == 0:
            save_checkpoint(global_model, optimizer, round + 1, args.checkpoints_dir)

if __name__ == "__main__":
    main()
