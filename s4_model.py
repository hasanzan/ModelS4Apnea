import torch
import torch.nn as nn
from s4 import S4Block as S4

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
  

class Encoder(nn.Module):
    r""" Initial Local Feature Extraction
    Args:
        in_chans (list(int)): Input channels
    """

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        # number of conv block
        num_blocks = len(in_channels) - 1

        # conv, pool, conv, pool
        self.layers = nn.ModuleList()

        # in channels
        out_channels = in_channels[1:]

        # create layers
        for i in range(num_blocks):
            conv = nn.Conv1d(in_channels[i], out_channels[i], kernel_size[i], padding="same")
            bn = nn.BatchNorm1d(out_channels[i])
            act = nn.ReLU()
            pool = nn.MaxPool1d(kernel_size=3, stride=2)
            
            self.layers.append(conv)
            self.layers.append(bn)
            self.layers.append(act)
            # self.layers.append(pool)

    def forward(self, x):
        """
        Input x is shape (B, C, L)
        """
        for layers in self.layers:
            x = layers(x)        
        return x


class S4Model(nn.Module):

    def __init__(
        self,
        in_channels,
        kernel_size,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001
    ):
        super().__init__()

        self.prenorm = prenorm

        # CNN encoder
        self.encoder = Encoder(in_channels, kernel_size)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True, bidirectional=False, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x