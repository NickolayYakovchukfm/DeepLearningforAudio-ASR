import torch
import torch.nn.functional as F
from torch import nn

from src.model.conformer_layers import ConformerBlock


class ConformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        n_tokens,
        enc_dim=256,
        heads=4,
        expansion_factor_conv=2,
        kernel_size=31,
        dropout_proba=0.1,
        expansion_factor_feedforward=4,
        conformer_blocks_num=4,
    ):
        super().__init__()

        self.linear_layer_1 = nn.Linear(input_dim, enc_dim)
        self.dropout = nn.Dropout(dropout_proba)

        self.conformer_blocks = nn.Sequential(
            *[
                ConformerBlock(
                    enc_dim,
                    heads,
                    expansion_factor_conv,
                    kernel_size,
                    dropout_proba,
                    expansion_factor_feedforward,
                )
                for _ in range(conformer_blocks_num)
            ]
        )

        self.linear_layer_2 = nn.Linear(enc_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **_):
        res = {}

        spec_features = spectrogram.transpose(1, 2)
        spec_features = self.linear_layer_1(spec_features)
        spec_features = self.conformer_blocks(spec_features)
        spec_features = self.linear_layer_2(spec_features)

        res["log_probs"] = F.log_softmax(spec_features, dim=-1)
        res["log_probs_length"] = spectrogram_length.clone()
        return res
