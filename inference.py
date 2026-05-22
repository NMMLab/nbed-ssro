import json

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download


class DiffractionResNet18(nn.Module):
    """
    ResNet18-based regression model for grayscale diffraction patterns.
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)

        self.model = resnet

    def forward(self, x):
        return self.model(x)


def load_model_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_model_from_checkpoint(weights_path, device="cpu"):
    """
    Load a model from a state_dict file.
    """
    model = DiffractionResNet18()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def download_model_from_hub(
    repo_id="NMMLab/nbed-ssro-model",
    weights_filename="model_state_dict.pt",
    config_filename="model_config.json",
):
    weights_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=weights_filename,
    )

    config_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=config_filename,
    )

    return weights_path, config_path


def preprocess_diffraction_array(X, config=None):
    """
    Preprocess diffraction intensities.

    Supported input shapes:
        (133, 133)
        (1, 133, 133)
        (N, 133, 133)
        (N, 1, 133, 133)

    Config options:
        log_clip_min: float or null
        normalize: bool
    """
    if config is None:
        config = {}

    log_clip_min = config.get("log_clip_min", None)
    normalize = config.get("normalize", False)

    X = torch.as_tensor(X, dtype=torch.float32)

    if log_clip_min is not None:
        X = torch.log(torch.clamp(X, min=float(log_clip_min)))

    if normalize:
        x_max = X.max()
        if x_max > 0:
            X = X / x_max

    if X.ndim == 2:
        X = X.unsqueeze(0).unsqueeze(0)
    elif X.ndim == 3:
        if X.shape[0] == 1:
            X = X.unsqueeze(0)
        else:
            X = X.unsqueeze(1)
    elif X.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported input shape: {tuple(X.shape)}")

    return X


def predict(model, X, device="cpu"):
    """
    Predict disorder parameter from diffraction patterns.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded model.
    X : torch.Tensor or array-like
        Input tensor of shape (N, 1, H, W) or compatible.
    device : str
        Device for inference.

    Returns
    -------
    numpy.ndarray
        Predicted target (1-disorder parameter), shape (N,)
    """
    X = torch.as_tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        target = model(X).detach().cpu().numpy().reshape(-1)

    return target


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path, config_path = download_model_from_hub(
        repo_id="NMMLab/nbed-ssro-model",
        weights_filename="model_state_dict.pt",
        config_filename="model_config.json",
    )

    config = load_model_config(config_path)

    model = load_model_from_checkpoint(
        weights_path=weights_path,
        device=device,
    )

    print("Model loaded successfully.")
    print(f"Weights: {weights_path}")
    print(f"Config: {config_path}")
    print(f"Config contents: {config}")
