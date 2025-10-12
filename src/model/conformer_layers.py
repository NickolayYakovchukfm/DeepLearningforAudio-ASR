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
        input_dim=256,
        kernel_size=32,
        dropout_proba=0.1,
        expansion_factor=2,
    ):
        super.__init__()
        padding_param = (kernel_size - 2) // 2

        self.layer_norm = nn.LayerNorm(input_dim)
        self.conv_1 = nn.Conv1d(input_dim, input_dim * expansion_factor, 1)
        self.GLu = nn.GLU(1)
        self.conv_2 = nn.Conv1d(
            input_dim, input_dim, kernel_size, padding=padding_param
        )
        self.bn = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.conv_3 = nn.Conv1d(input_dim, input_dim, 1)
        self.dp = nn.Dropout(p=dropout_proba)

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
        input_dim=256,
        heads=4,
        dropout_proba=0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.multiheadattn = nn.MultiheadAttention(input_dim, heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_proba)

    def forward(self, x):
        """
        Input:
            x.shape (b, t, c)
        Output:
            x.shape (b, t, c)
        """
        x_connect = x
        x = self.layer_norm(x)
        x = self.multiheadattn(x, x, x)
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
        input_dim=256,
        expansion_factor=4,
        dropout_proba=0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear_1 = nn.Linear(input_dim, input_dim * expansion_factor)
        self.swish = nn.SiLU()
        self.dropout_1 = nn.Dropout(p=dropout_proba)
        self.linear_2 = nn.Linear(input_dim * expansion_factor, input_dim)
        self.dropout_2 = nn.Dropout(p=dropout_proba)

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
        input_dim=256,
        heads=4,
        expansion_factor_conv=2,
        kernel_size=32,
        dropout_proba=0.1,
        expansion_factor_feedforward=4,
    ):
        super().__init__()

        self.feedforward = FeedForward(
            input_dim, expansion_factor_feedforward, dropout_proba
        )
        self.multiheadselfattn = MultiHeadSelfAttn(input_dim, heads, dropout_proba)
        self.convmod = ConvolutionModule(
            input_dim, kernel_size, dropout_proba, expansion_factor_conv
        )
        self.layer_norm = nn.LayerNorm(input_dim)

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
