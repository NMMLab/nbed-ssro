from xml.parsers.expat import model

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_boxdim(filename: str, line_offset: int = 5) -> np.ndarray:
    """
    Extract box dimensions from a LAMMPS data file.

    Parameters
    ----------
    filename : str
        Path to the LAMMPS data file.
    line_offset : int, optional
        Line offset to start reading box dimensions, by default 5.

    Returns
    -------
    numpy.ndarray
        Box dimension in shape (3, 2).
    """
    with open(filename, "r") as f:
        boxdim_lines = f.readlines()[line_offset : line_offset + 3]
    return np.array([list(map(float, line.split(" ")[:2])) for line in boxdim_lines])


def compute_loss_no_grad(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    criterion: nn.Module = nn.L1Loss(),
):
    """
    Compute loss without gradient calculation.
    """
    with torch.no_grad():
        model.eval()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = (
                X_batch.to(device),
                y_batch.to(device),
            )
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)
