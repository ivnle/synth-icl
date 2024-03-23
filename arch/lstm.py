import torch
import torch.nn as nn


class LSTMConfig:
    block_size: int = 1024
    vocab_size: int = -1  # defined later by tokenizer
    n_layer: int = 2
    n_embd: int = 768
    hidden_dim: int = 1024
    dropout_rate: float = 0.65


class LSTMModel(nn.Module):
    def __init__(
        self,
        embedder,
        head,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        proj_size=0,
    ):
        super().__init__()

        self.embedder = embedder
        self.head = head
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

    def forward(self, x):
        output, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        return output
