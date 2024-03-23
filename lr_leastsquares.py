from task import GMMDataset, LRDataset
import torch
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
from collections import defaultdict
from einops import rearrange
from tqdm import tqdm
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

TRAIN = "train"
EVAL = "eval"
TEST = "test"


# class LogReg(torch.nn.Module):
#     def __init__(self, dim, num_classes):
#         super().__init__()
#         self.linear = torch.nn.Linear(dim, num_classes)

#     def forward(self, x):
#         return self.linear(x)


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


def main(dims, val_noise):
    cfg_data = {
        "_name_": "linear-regression",
        "curriculum": {
            "dims": {
                "start": dims,
                "end": dims,
                "inc": 1,
                "interval": 2000,
            },
            "points_train": {
                "start": 32,
                "end": 32,
                "inc": 2,
                "interval": 2000,
            },
            "points_val": {
                "start": 1024,
                "end": 1024,
                "inc": 2,
                "interval": 2000,
            },
        },
        "task": "linear_regression",
        "data": "gaussian",
        "task_kwargs": {},
        "n_dims": dims,
        "train_noise": 0,
        "val_noise": val_noise, 
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
    # print(OmegaConf.to_yaml(cfg))
    
    # print(cfg.data.num_xy_pairs_train)

    generators = build_rand_generators(cfg)

    # train_dataset = GMMDataset(
    #     num_xy_pairs=cfg.data.num_xy_pairs_train,
    #     generators=generators[TRAIN],
    #     seed=cfg.seed,
    #     cfg=cfg,
    # )
    dataset = LRDataset(
        split=TEST,
        generators=generators[TEST],
        seed=cfg.seed_test,
        config=cfg,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
    )
    # keep track of MSE for each subsequence length
    mses = np.zeros(11)
    iter = 0
    for batch in tqdm(dataloader, total=cfg.n_outter_iters):
        if iter == cfg.n_outter_iters:
            print("Max iters reached. Terminating.")
            break
        inputs = batch["inputs"][:,::2,:].numpy()  # [1, 1024, d]
        targets = batch["inputs"][:,1::2,:].numpy()  # [1, 1024, d]
        targets = targets[:, :, 0] # [1, 1024]

        for _i in range(1, 11):
            # power of 2
            i = (2**_i) - 1
            input = inputs[:, :i, :]  # [1, i, d]
            input = rearrange(input, "b i d -> (b i) d")

            target = targets[:, :i]  # [1, i]
            target = rearrange(target, "b i -> (b i)")

            query_x = inputs[:, i, :]  # [1, d]
            query_y = targets[:, i]  # [1]

            model = Ridge(fit_intercept=False, alpha=0.01)
            # model = LinearRegression(fit_intercept=False)
            reg = model.fit(input, target)

            pred = model.predict(query_x)
            mse = mean_squared_error(pred, query_y)
            mses[_i] += mse

        iter += 1

    print(f"dims: {dims}, noise: {val_noise}")
    print("Final results:")
    nice_mses = mses / cfg.n_outter_iters
    # round to 3 decimal places
    nice_mses = np.round(nice_mses, 5)
    print(list(nice_mses))


if __name__ == "__main__":
    for dim in [5, 10, 20]:
        for noise in [0, 0.1, 0.5, 1]:
            main(dim, noise)
