import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
import numpy as np
import random
from einops import rearrange

IGNORE_IDX = -1


class ARDatasetSafari(Dataset):
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        tensor_example = self.tensor_dataset[index]

        dict_example = {
            "inputs": tensor_example[0],  # Assuming input is at index 0
            "targets": tensor_example[1]  # Assuming target is at index 1
            # You can add more keys here based on your requirements
        }

        return dict_example


class ARDataset(IterableDataset):
    def __init__(
        self, vocab_size: int, num_xy_pairs: int, generators: dict, seed: int, cfg
    ):
        super().__init__()
        self.vocab_size = vocab_size
        assert self.vocab_size % 2 == 0
        self.num_xy_pairs = num_xy_pairs
        self.seed = seed
        self.generators = generators
        self.vocab = list(range(vocab_size))
        self.cfg = cfg

    def build_key(self):
        shuf_vocab = self.generators["numpy"].permutation(self.vocab)
        keys, values = shuf_vocab[::2], shuf_vocab[1::2]
        return keys, values

    def build_sequence(self, keys, values):
        while True:
            # create a sequence
            key_idxs = self.generators["numpy"].choice(
                self.vocab_size // 2, self.num_xy_pairs
            )
            keys_seq = keys[key_idxs]
            values_seq = values[key_idxs]

            if self.cfg.data.force_target_in_prompt:
                # make sure the last key appears at least twice
                last_key = keys_seq[-1]
                last_key_count = np.count_nonzero(keys_seq == last_key)
                if last_key_count == 1:
                    continue
            # interleave keys and values
            sequence = rearrange([keys_seq, values_seq], "b n -> (n b)")
            break

        return sequence

    def build_inputs_targets(self, sequence):
        inputs = sequence[:-1]  # [s*2-1]
        targets = sequence[1::2]  # [s]
        # targets[1::2] = IGNORE_IDX
        return inputs, targets

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def __iter__(self):
        while True:
            keys, values = self.build_key()
            sequence = self.build_sequence(keys, values)
            inputs, targets = self.build_inputs_targets(sequence)
            yield {"inputs": inputs, "targets": targets}


def merge_cat(a, b):
    x = torch.cat((a, b), dim=-1)
    return x


class MergeEmbedder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.train.merge_type == "sum":
            self.embedder = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
            self.merge = torch.add
        elif cfg.train.merge_type == "concat":
            self.embedder = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model // 2)
            self.merge = merge_cat
        else:
            raise NotImplementedError

    def forward(self, x):
        # x [b, 2s, d], where s is number of in-context pairs
        x = self.embedder(x)

        # merge in-context example pairs
        a = x[:, ::2, :]
        b = x[:, 1::2, :]
        # mask the last element of b by setting to zero
        # b[:, -1, :] = 0

        bsz, s, d = b.shape
        # b is going to be one element shorter than a, so we pad it with zeros
        b_pad = torch.zeros(bsz, 1, d, device=b.device)
        b = torch.cat((b, b_pad), dim=1)

        x = self.merge(a, b)  # [b, s, d]
        return x


class AREmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.train.merge_embeds:
            self.embedder = MergeEmbedder(cfg)
        else:
            self.embedder = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)

    def forward(self, batch):
        return self.embedder(batch["inputs"])


class ARHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.model.d_model, cfg.data.vocab_size)

    def forward(self, x):
        return self.linear(x)


def ar_loss_fn(logits, targets, batch):
    # take loss on final token
    logits = logits[:, -1, :]  # [b, c]
    targets = targets[:, -1]  # [b]
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = rearrange(loss, "b -> b 1")
    return loss


def ar_loss_fn_parallel(logits, targets, batch, merge_embeds=False):
    # logits [b, (s*2)-1, d]
    # targets [b, s]
    batch_size = logits.shape[0]
    if not merge_embeds:
        # take loss on every other token
        logits = logits[:, ::2, :]  # [b, s, c]
    logits = rearrange(logits, "b s c -> (b s) c")
    targets = rearrange(targets, "b s -> (b s)")
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = rearrange(loss, "(b s) -> b s", b=batch_size)
    return loss
