import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np

expr2runs = {
    "l4_4b": [
        "src2uox4",
        "cgoaldfo",
        "5da9mlk8",
        "of0mlpzi",
        "co7e1q9r",
        "bwdbfmu4",
        "nuivffvh",
        "o4q6eoiw",
        "0tfmej3j",
        "qfnais5l",
        "v2j9o2l5",
        "htru6kfi",
    ],
    "1m_4b": [
        "kjetgfg9",
        "vr4nz6oc",
        "61zsns4i",
        "xw00djpi",
        "w4yqdktq",
        "n7u5jmbt",
        "6z6lqmyw",
        "u4dt8ao8",
        "src2uox4",  # llama2
        "of0mlpzi",  # retnet
        "ye2bi8tp",  # dynamicconv
        "p4twzhcp",  # h3
    ],
    "batch_4b": [
        "7xrprpzq",
        "bvshjgay",
        "gceifsma",
        "s4rg0r83",
        "src2uox4",
    ],
    "l4_12b": [],
    "1m_12b": [],
}


def compute_icl_score(runs, split="eval", tok_idxs=[49, 50, 51], tok_offset=450):
    run_ids = expr2runs[runs]
    arch2score = {}
    arch2std = {}
    arch2loss = {}
    for run_id in run_ids:
        fp = Path("/graft1/checkpoints/ivanlee/icl-arch", run_id)
        losses = Path(fp, f"per_token_loss_{split}.jsonl")
        df = pd.read_json(losses, lines=True)
        print(f"len(df): {len(df)}")
        # assert len(df) == 26

        # get arch type
        cfg = Path(fp, "cfg.yaml")
        # load yaml with omegaconf
        cfg = OmegaConf.load(cfg)

        if "batch" in runs:
            label = cfg.train.batch_size
        else:
            label = cfg.model._name_
            if label == "safari":
                label = cfg.model.layer._name_

        last_row = df.iloc[-1]
        avg_loss = np.mean(last_row["per_token_loss"])
        arch2loss[label] = avg_loss

        scores = []
        for tok_idx in tok_idxs:
            loss_t50 = last_row["per_token_loss"][tok_idx]
            loss_t500 = last_row["per_token_loss"][tok_idx + tok_offset]
            icl_score = loss_t500 - loss_t50
            scores.append(icl_score)

        # average the scores
        scores = np.array(scores)
        arch2score[label] = np.mean(scores)
        # gegt standard deviation of scores
        arch2std[label] = np.std(scores)

    # sort by icl score
    arch2score = {k: v for k, v in sorted(arch2score.items(), key=lambda item: item[0])}
    # convert dict into something i can paste into a spreadsheet after printing
    for k, v in arch2score.items():
        print(f"{k} {v:.4f} {arch2std[k]:.4f} {arch2loss[k]:.4f}")
    return arch2score


def main():
    run_ids = "l4_4b"
    # run_ids = "1m_4b"
    # run_ids = "batch_4b"
    tok_idxs = range(10, 30)
    print(len(tok_idxs))
    print(list(tok_idxs))
    compute_icl_score(run_ids, tok_idxs=tok_idxs, tok_offset=450)


if __name__ == "__main__":
    main()
