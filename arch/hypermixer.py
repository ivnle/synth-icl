import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x


class HyperMixerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.activation = nn.GELU()  # GELU for now, activation if activation else
        # self.h1 = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
        # self.h2 = None if cfg.tied else MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.feature_mixer = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
        # self.is_causal = cfg.is_causal

        if cfg.token_mixer == "fake_attention":
            self.h1 = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
            self.h2 = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
            self.token_mixer = self.hypermixer_attn_token_mixer
        elif cfg.token_mixer == "mlpmixer_causal":
            self.h1 = nn.Parameter(torch.randn(cfg.max_seq_len, cfg.max_seq_len))
            self.token_mixer = self.mlpmixer_causal_token_mixer
        elif cfg.token_mixer == "mlpmixer":
            self.h1 = nn.Linear(cfg.max_seq_len, cfg.d_inner)
            self.h2 = nn.Linear(cfg.d_inner, cfg.max_seq_len)
            self.token_mixer = self.mlpmixer_token_mixer
        elif cfg.token_mixer == "hypermixer":
            self.h1 = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
            self.h2 = MLP(cfg.d_model, cfg.d_inner, cfg.d_model)
            self.token_mixer = self.hypermixer_token_mixer
        elif cfg.token_mixer == "hypermixer_causal":
            self.h1 = MLP(cfg.d_model, cfg.d_inner, cfg.max_seq_len)
            self.token_mixer = self.hypermixer_causal_token_mixer
        else:
            raise ValueError(f"token_mixer {cfg.token_mixer} not supported")

    def hypermixer_attn_token_mixer(self, x):
        # x [b, s, m]
        W1 = self.h1(x)  # [b, s, n]
        W2 = self.h2(x)  # [b, s, n]
        W3 = torch.einsum("bzn, bsn -> bzs", W1, W2)
        causal_mask = torch.tril(torch.ones_like(W3), diagonal=0)
        W3 = W3 * causal_mask
        x = torch.einsum("bzs, bsm -> bzm", W3, x)
        x = F.gelu(x)
        x = self.norm(x)
        return x

    def mlpmixer_attn_token_mixer(self, x):
        pass

    def mlpmixer_causal_token_mixer(self, x):
        # x [b, s, m]
        # h1 [s, s]
        causal_mask = torch.tril(torch.ones_like(self.h1), diagonal=0)
        W = self.h1 * causal_mask
        x = torch.einsum("sz, bzm -> bsm", W, x)
        x = F.gelu(x)
        x = self.norm(x)
        return x

    def mlpmixer_token_mixer(self, x):
        # x [b, s, m]
        x = rearrange(x, "b s m -> b m s")
        x = self.h1(x)  # [b, m, n]
        x = F.gelu(x)
        x = self.h2(x)  # [b, m, s]
        x = rearrange(x, "b m s -> b s m")
        x = self.norm(x)
        return x

    def hypermixer_token_mixer(self, x):
        # x [b, s, m]
        W1 = self.h1(x)  # [b, s, n]
        W2 = self.h2(x)  # [b, s, n]
        W1 = rearrange(W1, "b s n -> b n s")
        P = torch.einsum("bns, bsm -> bnm", W1, x)
        A = F.gelu(P)
        output = torch.einsum("bsn, bnm -> bsm", W2, A)
        output = A
        output = self.norm(output)
        return output

    def hypermixer_causal_token_mixer(self, x):
        # x [b, s, m]
        W = self.h1(x)  # [b, s, z]
        W = rearrange(W, "b s z -> b z s")
        causal_mask = torch.tril(torch.ones_like(W), diagonal=0)
        W = W * causal_mask
        x = torch.einsum("bzs, bsm -> bzm", W, x)
        x = F.gelu(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x_1 = x + self.token_mixer(self.norm(x))
        x_out = x + self.feature_mixer(self.norm(x_1))

        return x_out


class HyperMixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([HyperMixerLayer(cfg) for _ in range(cfg.n_layer)])
        self.pos_embed = nn.Embedding(
            num_embeddings=cfg.max_seq_len, embedding_dim=cfg.d_model
        )

    def forward(self, x):
        seq_len = x.shape[1]
        pos = self.pos_embed(torch.arange(seq_len, device=x.device))
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        return x


class HyperMixerModel(nn.Module):
    def __init__(self, cfg, embedder, head):
        super().__init__()
        self.model = HyperMixer(cfg)
        self.embedder = embedder
        self.head = head

    def forward(self, x):
        output = self.model(x)
        return output
