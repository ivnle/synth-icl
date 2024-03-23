from task import GMMDataset
import torch
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
from collections import defaultdict
from einops import rearrange
from tqdm import tqdm

TRAIN = "train"
EVAL = "eval"
TEST = "test"


class LogReg(torch.nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def build_rand_generators(cfg):
    gs = defaultdict(dict)
    modules = ["torch", "numpy", "random"]
    split2seed = {
        TRAIN: cfg.seed,
        # EVAL_TRAIN: cfg.seed,
        EVAL: cfg.seed_eval,
        TEST: cfg.seed_test,
    }

    for split, seed in split2seed.items():
        for module in modules:
            if module == "torch":
                g = torch.Generator()
                g.manual_seed(seed)
            elif module == "numpy":
                g = np.random.default_rng(seed)
            elif module == "random":
                g = random.Random(seed)
            else:
                raise ValueError(f"Unknown module: {module}")
            gs[split][module] = g

    return gs


def main():
    cfg_data = {
        "num_xy_pairs_val": 1024,
        # "num_xy_pairs_val": 10,
        "num_classes": 2,
        "dim": 16,
    }
    cfg = {
        "data": cfg_data,
        "seed": 1000,
        "seed_eval": 2000,
        "seed_test": 3000,
        "n_outter_iters": 1000,
        # "n_outter_iters": 2,
        "n_inner_iters": 1000,
        "lr": 0.01,
    }
    cfg = DictConfig(cfg)
    # nicely print cfg
    print(OmegaConf.to_yaml(cfg))
    # print(cfg.data.num_xy_pairs_train)

    generators = build_rand_generators(cfg)

    # train_dataset = GMMDataset(
    #     num_xy_pairs=cfg.data.num_xy_pairs_train,
    #     generators=generators[TRAIN],
    #     seed=cfg.seed,
    #     cfg=cfg,
    # )
    dataset = GMMDataset(
        num_xy_pairs=cfg.data.num_xy_pairs_val,
        generators=generators[TEST],
        seed=cfg.seed_test,
        cfg=cfg,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
    )
    # keep track of accuracy for each subsequence length
    accs = np.zeros(11)
    iter = 0
    for batch in tqdm(dataloader, total=cfg.n_outter_iters):
        if iter == cfg.n_outter_iters:
            print("Max iters reached. Terminating.")
            break
        inputs = batch["inputs"]  # [1, 1024, 16]
        inputs = inputs.to("cuda")
        targets = batch["targets"]  # [1, 1024]
        targets = targets.to("cuda")
        class_vectors = batch["class_vectors"]

        for _i in range(0, 11):
            # power of 2
            i = (2**_i) - 1
            input = inputs[:, :i, :]  # [1, i, 16]
            input = rearrange(input, "b i d -> (b i) d")

            target = targets[:, :i]  # [1, i]
            target = rearrange(target, "b i -> (b i)")

            query = inputs[:, i, :]  # [1, 16]
            query_target = targets[:, i]  # [1]

            # print(f"{input.shape=}")
            # print(f"{target.shape=}")
            # print(f"{query.shape=}")
            # print(f"{query_target.shape=}")
            # foo

            # train a logistic regression model on each subsequence
            model = LogReg(cfg.data.dim, cfg.data.num_classes)
            model = model.to("cuda")
            # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(cfg.n_inner_iters):
                optimizer.zero_grad()
                logits = model(input)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

            logits = model(query)
            pred = torch.argmax(logits, dim=-1)
            acc = torch.sum(pred == query_target).item()
            accs[_i] += acc

            # print(f"{query=}")
            # print(f"{query_target=}")
            # print(f"{pred=}")
            # print(f"{i=}, {acc=}")
        # foo
        if iter % 20 == 0:
            nice_accs = accs / (iter + 1)
            # round to 3 decimal places
            nice_accs = np.round(nice_accs, 2)
            print(nice_accs)
        iter += 1

    print("Final results:")
    nice_accs = accs / cfg.n_outter_iters
    # round to 3 decimal places
    nice_accs = np.round(nice_accs, 3)
    print(nice_accs)


if __name__ == "__main__":
    main()
