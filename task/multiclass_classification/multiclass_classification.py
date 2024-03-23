import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from einops import rearrange, reduce
import numpy as np
import random

IGNORE_IDX = -1


def convert_uniform(x):
    # convert from U(0, 1) to U(-1, 1)
    return x * 2 - 1


class MCCDataset(IterableDataset):
    def __init__(self, num_xy_pairs, generators: dict, seed: int, cfg):
        super().__init__()
        self.num_classes = cfg.data.num_classes
        self.num_xy_pairs = num_xy_pairs
        self.generators = generators
        self.dim = cfg.data.dim
        self.seed = seed
        self.cfg = cfg

        if cfg.data.x_distr == "normal":
            self.x_sampler = torch.randn
        elif cfg.data.x_distr == "uniform":
            self.x_sampler = torch.rand
        else:
            raise ValueError(f"Unknown x_distr: {cfg.data.x_distr}")

        if cfg.data.y_distr == "normal":
            self.y_sampler = torch.randn  # N(0, 1)
        elif cfg.data.y_distr == "uniform":
            self.y_sampler = torch.rand  # U(0, 1)
        else:
            raise ValueError(f"Unknown y_distr: {cfg.data.y_distr}")

        # if cfg.data.do_gmm:
        #     self.__iter__ = self.gmm_iter
        # else:
        #     self.__iter__ = self.iter

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def __iter__(self):
        while True:
            # each class corresponds to a random vector
            class_vectors = self.y_sampler(
                (self.num_classes, self.dim), generator=self.generators["torch"]
            )
            if self.cfg.data.y_distr == "uniform":
                class_vectors = convert_uniform(class_vectors)
            # the class of an input is determined by the class vector with
            # the highest dot product with the input
            x = self.x_sampler(
                (self.num_xy_pairs, self.dim), generator=self.generators["torch"]
            )
            if self.cfg.data.x_distr == "uniform":
                x = convert_uniform(x)

            scores = x @ class_vectors.T
            y = torch.argmax(scores, dim=1)

            yield {"inputs": x, "targets": y, "class_vectors": class_vectors}


class GMMDataset(IterableDataset):
    def __init__(self, num_xy_pairs, generators: dict, seed: int, cfg):
        super().__init__()
        self.num_classes = cfg.data.num_classes
        self.num_xy_pairs = num_xy_pairs
        self.generators = generators
        self.dim = cfg.data.dim
        self.seed = seed
        self.cfg = cfg

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def __iter__(self):
        while True:
            # sample k vectors from U[0, 1]
            class_vectors = torch.rand(
                self.num_classes, self.dim, generator=self.generators["torch"]
            )
            # convert to U(-1, 1)
            class_vectors = convert_uniform(class_vectors)
            # randomly choose seqeunce of classes
            y = torch.randint(
                low=0,
                high=self.num_classes,
                size=(self.num_xy_pairs,),
                generator=self.generators["torch"],
            )
            u = class_vectors[y]
            # sample each x from N(0, 1)
            x = torch.randn(
                self.num_xy_pairs, self.dim, generator=self.generators["torch"]
            )
            # update each x such that was sampled from N(u, 1)
            x = x + u

            yield {"inputs": x, "targets": y, "class_vectors": class_vectors}


def merge_cat(a, b):
    x = torch.cat((a, b), dim=-1)
    return x


class MergeEmbedder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.train.merge_type == "sum":
            self.embedder = torch.nn.Linear(cfg.data.dim, cfg.model.d_model)
            self.merge = torch.add
        elif cfg.train.merge_type == "concat":
            self.embedder = torch.nn.Linear(cfg.data.dim, cfg.model.d_model // 2)
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
        b[:, -1, :] = 0
        x = self.merge(a, b)  # [b, s, d]
        return x


class MCCEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.train.merge_embeds:
            self.embedder = MergeEmbedder(cfg)
        else:
            self.embedder = nn.Linear(cfg.data.dim, cfg.model.d_model)
        self.y_embedder = nn.Embedding(cfg.data.num_classes, cfg.data.dim)
        self.merge_embeds = cfg.train.merge_embeds

    def forward(self, batch):
        x = batch["inputs"]  # [b, s, d]
        y = batch["targets"]  # [b, s]

        y = self.y_embedder(y)  # [b, s, d]

        # interleave x and y
        inputs = rearrange([x, y], "t b s d -> b (s t) d")
        inputs = self.embedder(inputs)
        if not self.merge_embeds:
            # drop the last y
            inputs = inputs[:, :-1, :]  # [b, (s*2)-1, d]
        return inputs


class MCCHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.model.d_model, cfg.data.num_classes)

    def forward(self, x):
        return self.linear(x)


def mcc_loss_fn(logits, targets, batch):
    # take loss on final token
    logits = logits[:, -1, :]  # [b, c]
    targets = targets[:, -1]  # [b]
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = rearrange(loss, "b -> b 1")
    return loss


def mcc_loss_fn_parallel(logits, targets, batch, merge_embeds=False):
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
