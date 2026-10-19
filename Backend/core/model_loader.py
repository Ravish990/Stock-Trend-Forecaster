from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import torch

from Backend.core.config import MODEL_PATH, SCALER_PATH
from Model.model import LSTMModel
from Model.train_model import FEATURES

logger = logging.getLogger(__name__)


def load_model_and_scaler(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    device: str = "cpu",
) -> Tuple[LSTMModel, any]:
    """
    Loads and validates the trained PyTorch LSTM checkpoint and StandardScaler.
    Sets model to evaluation mode.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    scaler = joblib.load(scaler_path)

    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=32,
        num_layers=2,
        dropout=0.5,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    logger.info("Successfully loaded PyTorch LSTM model and scaler.")
    return model, scaler

