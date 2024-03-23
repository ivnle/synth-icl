import torch
import torch.nn.functional as F
from einops import reduce, rearrange
from .curriculum import Curriculum
from .tasks import get_task_sampler
from .samplers import get_data_sampler
from random import randint
import numpy as np
import random


class LRDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, generators, seed, config):
        self.curriculum = Curriculum(config.data.curriculum, split)
        self.data_sampler = get_data_sampler(
            config.data.data, n_dims=config.data.n_dims
        )
        num_tasks = config.data.num_tasks if "num_tasks" in config.data else None

        self.task_sampler = get_task_sampler(
            config.data.task,
            config.data.n_dims,
            1,
            num_tasks=num_tasks,
            **config.data.task_kwargs,
        )
        self.config = config
        self.split = split
        self.seed = seed
        self.generators = generators
        self.gauss_noise_std = config.data.train_noise if split == "train" else config.data.val_noise

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def sample_seeds(self, total_seeds, count):
        seeds = set()
        while len(seeds) < count:
            seeds.add(randint(0, total_seeds - 1))
        return seeds

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(points, 1),
                torch.zeros(points, dim - 1, device=ys_b.device),
            ),
            axis=1,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=1)
        zs = zs.view(2 * points, dim)
        return zs

    def __iter__(self):
        while True:
            data_sampler_args = {}
            task_sampler_args = {}
            if "sparse" in self.config.data.task:
                task_sampler_args["valid_coords"] = self.curriculum.n_dims_truncated
            if "num_training_examples" in self.config:
                assert (
                    self.config.data.num_training_examples
                    >= self.config.train.batch_size
                )
                seeds = self.sample_seeds(
                    self.config.data.num_training_examples,
                    self.config.train.batch_size,
                )
                data_sampler_args["seeds"] = seeds
                task_sampler_args["seeds"] = [s + 1 for s in seeds]

            task_sampler_args["generators"] = self.generators
            xs = self.data_sampler.sample_xs(
                self.curriculum.n_points,
                1,
                self.generators,
                self.curriculum.n_dims_truncated,
                **data_sampler_args,
            ).squeeze()
            task = self.task_sampler(**task_sampler_args)
            ys = task.evaluate(xs).squeeze()

            # add guassian noise to ys
            noisy_ys = ys + self.gauss_noise_std * torch.randn_like(ys)

            zs = self._combine(xs, noisy_ys)
            yield {"inputs": zs, "targets": ys}


def merge_sum(a, b):
    return a + b


def merge_cat(a, b):
    x = torch.cat((a, b), dim=-1)
    return x


class MergeEmbedder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.train.merge_type == "sum":
            self.embedder = torch.nn.Linear(cfg.data.n_dims, cfg.model.d_model)
            self.merge = torch.add
        elif cfg.train.merge_type == "concat":
            self.embedder = torch.nn.Linear(cfg.data.n_dims, cfg.model.d_model // 2)
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

        # bsz, s, d = b.shape
        # # b is going to be one element shorter than a, so we pad it with zeros
        # b_pad = torch.zeros(bsz, 1, d, device=b.device)
        # b = torch.cat((b, b_pad), dim=1)

        x = self.merge(a, b)  # [b, s, d]
        return x


class LREmbedder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.train.merge_embeds:
            self.embedder = MergeEmbedder(cfg)
        else:
            self.embedder = torch.nn.Linear(cfg.data.n_dims, cfg.model.d_model)

    def forward(self, batch: dict[str, torch.Tensor]):
        inputs = batch["inputs"]
        return self.embedder(inputs)


class LRHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head = torch.nn.Linear(cfg.model.d_model, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.head(x)


def lr_loss_fn(logits, targets, batch):
    # TODO: this is a hack to make perm invar experiments work
    # need to go back to -2 instead of -1 later
    logits = logits[:, -1, 0]  # [b]
    # logits = logits[:, -2, 0]  # [b]
    targets = targets[:, -1]  # [b]
    loss = F.mse_loss(logits, targets, reduction="none")  # [b]
    loss = rearrange(loss, "b -> b 1")
    return loss


def lr_loss_fn_parallel(logits, targets, batch, merge_embeds=False):
    # logits [b, (s*2)-1, d]
    # targets [b, s]
    logits = logits[:, ::2, 0]  # [b, s]
    loss = F.mse_loss(logits, targets, reduction="none")  # [b, s]
    return loss
