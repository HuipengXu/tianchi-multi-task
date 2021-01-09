from typing import Dict, NamedTuple, Optional

import numpy as np
import torch


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used
    to compute metrics.
    """

    predictions: np.ndarray
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Dict[str, torch.Tensor]
    task_probs: Dict[str, torch.Tensor]
    label_ids: Optional[Dict[str, torch.Tensor]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


PREFIX_CHECKPOINT_DIR = "checkpoint"
