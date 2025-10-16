import audiomentations
import numpy as np
import torch
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = audiomentations.AddGaussianNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        x = data.unsqueeze(1)
        return self._aug(x, sample_rate=16000).squeeze(1)
