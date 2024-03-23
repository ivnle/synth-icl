import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(
        self,
        embedder,
        head,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
    ):
        super().__init__()

        self.embedder = embedder
        self.head = head
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        output, _ = self.gru(x)  # [batch_size, seq_len, hidden_dim]
        return output
