import torch
from torch import nn

DIVIDE_FFN_CONST = 0.5


class ConvolutionModule(nn.Module):
    """
    ^ -> LayerNorm -> PWConv -> GLU -> 1D_DWConv -> BN -> Swish -> PWConv -> Dropout -> ^ +
    The convolution module contains a pointwise convolution with an expansion factor of 2 projecting the
    number of channels with a GLU activation layer, followed by a 1-D Depthwise convolution. The 1-D depthwise conv is followed by a
    Batchnorm and then a swish activation layer.
    """

    def __init__(
        self,
        input_dim,
        kernel_size,
        dropout_proba=0.1,
        expansion_factor=2,
    ):
        super.__init__()

        self.layer_norm = nn.LayerNorm()
        self.conv_1 = nn.Conv1d()
        self.GLu = nn.GLU()
        self.conv_2 = nn.Conv1d()
        self.bn = nn.BatchNorm1d()
        self.swish = nn.SiLU()
        self.conv_3 = nn.Conv1d()
        self.dp = nn.Dropout()

    def forward(self, x):
        """
        Input:
            x.shape (b, t, c)
        Output:
            x.shape (b, t, c)
        """
        x_connect = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.conv_1(x)
        x = self.GLu(x)
        x = self.conv_2(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv_3(x)
        x = self.dp(x)
        x = x.transpose(1, 2)
        x += x_connect

        return x


class MultiHeadSelfAttn(nn.Module):
    """
    ^ -> LayerNorm -> MultiHeadAttn with Pos Emb / rotary -> Dropuot -> ^ +
    We use multiheaded self-attention with relative positional embedding in a
    pre-norm residual unit.
    """

    def __init__(
        self,
        input_dim,
        heads,
        enc_len,
        dropout_proba=0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm()
        self.multiheadattn = nn.MultiheadAttention()
        self.dropout = nn.Dropout()

    def forward(self, x):
        """
        Input:
            x.shape (b, t, c)
        Output:
            x.shape (b, t, c)
        """
        x_connect = x
        x = self.layer_norm(x)
        x = self.multiheadattn(x)
        x = self.dropout(x)
        x += x_connect

        return x


class FeedForward(nn.Module):
    """
    ^ -> LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout -> ^ +
    The first linear layer uses an expansion factor of 4 and the second linear layer projects it back to the
    model dimension. We use swish activation and a pre-norm residual units in feed forward module.
    """

    def __init__(
        self,
        input_dim,
        expansion_factor=4,
        dropout_proba=0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm()
        self.linear_1 = nn.Linear()
        self.swish = nn.SiLU()
        self.dropout_1 = nn.Dropout()
        self.linear_2 = nn.Linear()
        self.dropout_2 = nn.Dropout()

    def forward(self, x):
        x_connect = x
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.swish(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        x += x_connect

        return x


class ConformerBlock(nn.Module):
    """
    x + 1/2 FFN -> + MHSA -> + Conv -> LayerNorm
    """

    def __init__(
        self,
        input_dim,
        heads,
        expansion_factor_conv,
        kernel_size,
        expansion_factor,
        dropout_proba,
        enc_len,
        expansion_factor_feedforward,
    ):
        super().__init__()

        self.feedforward = FeedForward()
        self.multiheadselfattn = MultiHeadSelfAttn()
        self.convmod = ConvolutionModule()
        self.layer_norm = nn.LayerNorm()

    def forward(self, x):
        """
        Input:
            x.shape (b, t, c)
        Output:
            x.shape (b, t, c)
        """
        x += DIVIDE_FFN_CONST * self.feedforward(x)
        x += self.multiheadselfattn(x)
        x += self.convmod(x)
        x = self.layer_norm(x + DIVIDE_FFN_CONST * self.feedforward(x))

        return x
