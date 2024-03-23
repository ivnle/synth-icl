# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

# Load the JSON lines file into a DataFrame
fp = "/trunk/ivanlee/icl-arch/qui5jg1k/per_token_loss_eval.jsonl"
df = pd.read_json(fp, lines=True)

# Set up the log scales for the axes
plt.yscale("log")
plt.xscale("log", base=2)

# Create an array of indices corresponding to powers of 2
max_index = int(math.log(len(df), 2)) + 1
power_of_2_indices = [2**i for i in range(max_index)]

# Plot each row as a line in the graph
for i in range(len(df)):
    row = df.iloc[i]
    y = row["per_token_loss"]

    # Only plot the indices that are powers of 2
    y_power_of_2 = [y[i] for i in power_of_2_indices if i < len(y)]

    # Plot the data
    plt.plot(
        power_of_2_indices[: len(y_power_of_2)], y_power_of_2, label=row["train_iter"]
    )

# Show legend
plt.legend()

# Set x ticks to powers of 2
plt.xticks(power_of_2_indices)

# Set x and y axis labels
plt.xlabel("Index (Powers of 2)")
plt.ylabel("Per Token Loss")

# Set title
plt.title("Per Token Loss Evaluation")

# Show the plot
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from pathlib import Path

# run_id = 'aa465fzq' # h3
# run_id = 'f0p4n5yu' # hyena
# run_id = 'th2iimmk' # s4
run_id = "5mgl365d"  # transformer

# Load the JSON lines file into a DataFrame
fp = Path("/trunk/ivanlee/icl-arch", run_id, "per_token_loss_eval.jsonl")
df = pd.read_json(fp, lines=True)

# Set up the log scales for the axes
plt.yscale("log")
plt.xscale("log", base=2)

# set max y value to 4
plt.ylim(1, 20)

# Create an array of indices corresponding to powers of 2
n_tokens = len(df.iloc[0]["per_token_loss"])
max_index = int(math.log(n_tokens, 2)) + 1
power_of_2_indices = [2**i for i in range(max_index)]

# Create a colormap for the gradient effect
cmap = plt.get_cmap("Blues")  # You can choose any colormap you like

# Normalize the row indices to [0, 1]
normalize = plt.Normalize(0, len(df) - 1)

# Plot each row as a line in the graph
for i in range(len(df)):
    row = df.iloc[i]
    y = row["per_token_loss"]

    # Only plot the indices that are powers of 2
    y_power_of_2 = [y[i - 1] for i in power_of_2_indices]

    # Calculate the color based on the normalized row index
    color = cmap(normalize(i))

    # Plot the data with the calculated color
    plt.plot(
        power_of_2_indices[: len(y_power_of_2)],
        y_power_of_2,
        label=row["train_iter"],
        color=color,
    )

# Show legend
# plt.legend()

# Set x ticks to powers of 2
plt.xticks(power_of_2_indices)

# Set x and y axis labels
plt.xlabel("Token Index")
plt.ylabel("Loss")

# Set title
plt.title("Evaluation")

# Show the plot
plt.show()

# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def plot_loss_vs_token_idx(run_ids: list[str], split: str = "eval"):
    """
    split: 'eval' or 'train'
    """
    # Set up the log scales for the axes
    plt.yscale("log")
    plt.xscale("log", base=2)

    # set max y value to 4
    plt.ylim(1, 5)

    for run_id in run_ids:
        fp = Path("/trunk/ivanlee/icl-arch", run_id)
        losses = Path(fp, f"per_token_loss_{split}.jsonl")
        df = pd.read_json(losses, lines=True)

        # get arch type
        cfg = Path(fp, "cfg.yaml")
        # load yaml with omegaconf
        cfg = OmegaConf.load(cfg)

        arch = cfg.model._name_
        if arch == "safari":
            arch = cfg.model.layer._name_

        # Create an array of indices corresponding to powers of 2
        n_tokens = len(df.iloc[0]["per_token_loss"])
        max_index = int(math.log(n_tokens, 2)) + 1
        power_of_2_indices = [2**i for i in range(max_index)]

        # get the row that corresponds with the last eval
        row = df.iloc[-1]
        y = row["per_token_loss"]

        # Only plot the indices that are powers of 2
        y_power_of_2 = [y[i - 1] for i in power_of_2_indices]

        # Plot the data with the calculated color
        plt.plot(
            power_of_2_indices[: len(y_power_of_2)],
            y_power_of_2,
            # label=row["train_iter"],
            label=arch,
        )

    # Show legend
    plt.legend()

    # Set x ticks to powers of 2
    plt.xticks(power_of_2_indices)

    # Set x and y axis labels
    plt.xlabel("Token Index")
    plt.ylabel("Loss")

    # Set title
    plt.title("Evaluation")

    # Show the plot
    plt.show()


runs = [
    "5mgl365d",  # llama2
    "n3acpadw",  # s4
    "nr9avy1t",  # hyena
    "wshkpb8t",  # gpt2
    "hcqxvzqi",  # h3
    "jsvaom7b",  # lstm
    # "uvkwuyfg", # transformer
    # '91qc34y4', # rwkv (crashed)
    "fy89udr2",  # rwkv (in progress)
]
plot_loss_vs_token_idx(runs, split="eval")

# %%
