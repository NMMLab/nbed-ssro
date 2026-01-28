import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from utils import get_boxdim, compute_loss_no_grad
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from tqdm import tqdm


class DiffractionDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initialize a dataset from diffraction patterns and environment descriptor(s).

        Parameters
        ----------
        X : torch.Tensor
            Diffraction pattern intensity data in shape (N, C, H, W)
        y : torch.Tensor
            Environment descriptor(s) in shape (N) or(N, d)
        """
        self.X = X
        self.y = y
        self.transform = v2.RandomApply(
            transforms=[
                v2.RandomRotation(
                    (-180.0, 180.0), interpolation=v2.InterpolationMode.BILINEAR
                ),
                v2.RandomAutocontrast(),
                v2.GaussianNoise(),
                v2.RandomAdjustSharpness(0.5),
            ],
            p=0.8,
        )
        self.augmentation = False

    def train(self):
        """
        Enable data augmentation.
        """
        self.augmentation = True

    def eval(self):
        """
        Disable data augmentation.
        """
        self.augmentation = False

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.augmentation:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.X[idx], self.y[idx]


from torchvision import models


class DiffractionCNN(nn.Module):
    def __init__(self, pre_trained: bool = False):
        super().__init__()
        # Load pretrained ResNet18
        if pre_trained:
            resnet = models.resnet18(weights="DEFAULT")
        else:
            resnet = models.resnet18()
        # Modify first conv layer to accept 1-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)
        self.model = resnet

    def forward(self, x):
        return self.model(x)


criterion = nn.L1Loss()


def _train_model_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    train_loader : DataLoader
        The data loader for the training dataset.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    device : torch.device
        The device to run the training on.
    device : torch.device, optional
        The device to run the training on, by default torch.device("cpu")

    Returns
    -------
    float
        The average training loss for the epoch.
    """
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = (
            X_batch.to(device),
            y_batch.to(device),
        )
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += torch.mean(loss).item()

    return train_loss / len(train_loader)


def train(
    folder: str,
    model: nn.Module,
    train_dataset: torch.utils.data.Subset[DiffractionDataset],
    val_dataset: torch.utils.data.Subset[DiffractionDataset],
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0,
    device: torch.device = torch.device("cpu"),
    print_interval: int | None = None,
    save_interval: int | None = None,
    save_patience: int = 0,
):
    """Train the model.

    Parameters
    ----------
    folder : str
        Folder to save checkpoints and loss logs.
    model : nn.Module
        The model to be trained.
    train_dataset : torch.utils.data.Subset[DiffractionDataset]
        The training dataset.
    val_dataset : torch.utils.data.Subset[DiffractionDataset]
        The validation dataset.
    num_epochs : int, optional
        Number of training epochs, by default 100
    batch_size : int, optional
        Batch size for training, by default 64
    lr : float, optional
        Learning rate, by default 0.001
    weight_decay : float, optional
        Weight decay for optimizer, by default 0
    device : torch.device, optional
        Device to run the training on, by default torch.device("cpu")
    print_interval : int | None, optional
        Interval (in epochs) to print training progress, by default None
    save_interval : int | None, optional
        Interval (in epochs) to save model checkpoints, by default None
    save_patience : int, optional
        Minimum number of epochs before saving checkpoints, by default 0

    Returns
    -------
    None
    """
    if not print_interval:
        print_interval = max(int(num_epochs / 10), 10)
    if not save_interval:
        save_interval = max(int(num_epochs / 10), 10)
    os.makedirs(folder, exist_ok=True)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loss = []
    val_loss = []

    with open(f"{folder}/loss.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = torch.inf
    best_epoch_idx = 0

    # Training
    for epoch in tqdm(range(num_epochs)):
        train_loader.dataset.dataset.train()
        train_loss_epoch = _train_model_epoch(model, train_loader, optimizer, device)
        train_loss.append(train_loss_epoch)
        val_loader.dataset.dataset.eval()
        model.eval()
        val_loss_epoch = compute_loss_no_grad(model, val_loader, device, criterion)
        scheduler.step(val_loss_epoch)
        val_loss.append(val_loss_epoch)

        with open(f"{folder}/loss.csv", "a") as f:
            f.write(
                ",".join(
                    [str(epoch + 1)]
                    + [f"{loss:.4f}" for loss in [train_loss_epoch, val_loss_epoch]]
                )
                + "\n"
            )
        if ((epoch + 1) % print_interval == 0) or (epoch == 0):
            print(
                f"Epoch {epoch}/{num_epochs}, "
                f"training Loss: {train_loss[-1]:.4f}, "
                f"validation Loss: {val_loss[-1]:.4f}, "
                f"lr: {scheduler.get_last_lr()[0]}"
            )
        if (
            ((epoch + 1) % save_interval == 0)
            and (best_val_loss - val_loss_epoch > 1e-4)
            and (epoch + 1 > save_patience)
        ):
            best_val_loss = val_loss_epoch
            best_epoch_idx = epoch
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss[-1],
                },
                f"{folder}/checkpoint_{epoch+1}.pt",
            )
        elif epoch + 1 == num_epochs:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": train_loss[-1],
                },
                f"{folder}/checkpoint_{epoch+1}.pt",
            )

    model.load_state_dict(
        torch.load(f"{folder}/checkpoint_{best_epoch_idx+1}.pt")["model_state_dict"]
    )

    model.eval()
    train_loader.dataset.dataset.eval()
    y_true_train = []
    y_pred_train = []
    y_true_val = []
    y_pred_val = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = (
                X_batch.to(device),
                y_batch.to(device),
            )
            y_true_train.append(y_batch)
            y_pred = model(X_batch)
            y_pred_train.append(y_pred)
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))
            y_true_val.append(y_batch)
            y_pred = model(X_batch)
            y_pred_val.append(y_pred)

    torch.save(
        {
            "y_true_train": torch.cat(y_true_train, dim=0),
            "y_pred_train": torch.cat(y_pred_train, dim=0),
            "y_true_val": torch.cat(y_true_val, dim=0),
            "y_pred_val": torch.cat(y_pred_val, dim=0),
        },
        f"{folder}/y_pred_dict_{best_epoch_idx+1}.pt",
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("X_data", type=str, help="Path to X_data.pt file")
    argparser.add_argument("y_data", type=str, help="Path to y_data.pt file")
    argparser.add_argument(
        "--lammps_datafile",
        type=str,
        default="../mcmd_aniso/data.minimized",
        help="Path to LAMMPS data file",
    )
    argparser.add_argument(
        "--saed_geometry_log",
        type=str,
        default="../data/saed_geometry_log.pkl",
        help="Path to saed_geometry_log.pkl file",
    )
    argparser.add_argument(
        "--boundary-threshold",
        type=float,
        default=5.0,
        help="Boundary threshold for filtering",
    )
    args = argparser.parse_args()

    X_data = torch.log(torch.clip(torch.load("../data/X_data.pt"), min=1.0)).view(
        -1, 1, 133, 133
    )
    y_data = torch.load("../data/y_avg_orderparam.pt").view(-1, 1)

    if args.lammps_datafile is not None:
        boxdims = get_boxdim(args.lammps_datafile)
        log_data = pd.read_pickle(args.saed_geometry_log)
        data_filter = (
            (log_data["axis"] == "y")
            & (log_data["upper_bound"] < boxdims[1, 1] - args.boundary_threshold)
            & (log_data["lower_bound"] > boxdims[1, 0] + args.boundary_threshold)
        ) | (
            (log_data["axis"] == "z")
            & (log_data["upper_bound"] < boxdims[2, 1] - args.boundary_threshold)
            & (log_data["lower_bound"] > boxdims[2, 0] + args.boundary_threshold)
        )

        X_data = X_data[data_filter].view(-1, 1, 133, 133)
        y_data = y_data[data_filter].view(-1, 1)

    X_data = X_data / X_data.max()

    print(f"X_data.shape = {X_data.shape}")
    print(f"y_data.shape = {y_data.shape}")
    print(f"X_data: min={X_data.min()}, max={X_data.max()}")
    print(f"y_data: min={y_data.min()}, max={y_data.max()}")

    dataset = DiffractionDataset(X_data, y_data)

    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device='{device}'")

    model = DiffractionCNN(pre_trained=True)

    train(
        folder="checkpoints",
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=400,
        batch_size=64,
        lr=0.001,
        weight_decay=1e-5,
        device=device,
        print_interval=10,
        save_interval=1,
        save_patience=10,
    )
