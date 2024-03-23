# %%
import pandas as pd
import random
import transformers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import jsonlines
import numpy as np


def build_sentiment_prompt(df, emotion, n_examples, same_name) -> list:
    _df = df[df["Class"] == emotion]
    rows = df.sample((n_examples), replace=False)
    samples = []
    for row in rows.iterrows():
        sample, _ = build_sentiment_sample(row[1], same_name)
        samples.append(sample)
    return samples


def build_sentiment_sample(row, same_name=None, is_query=False) -> tuple[str, str]:
    subject = row["Subject"]
    if same_name is not None:
        subject = same_name
    sentence = row["Sentence"].replace(row["Subject"], subject)
    target = row["Class"]
    if is_query:
        input = f"{sentence} {subject} is"
    else:
        input = f"{sentence} {subject} is {target}."
    return input, target


def build_sentiment_dataset(
    examples_per_class, same_name="Lilly", fp="/home/ivanlee/icl-arch/stats/sent.csv"
):
    df = pd.read_csv(fp)
    inputs, targets = [], []

    for i, row in enumerate(df.iterrows()):
        query, target = build_sentiment_sample(row[1], same_name, is_query=True)
        happy_prompt = build_sentiment_prompt(
            df.drop(i), "happy", examples_per_class, same_name
        )
        sad_prompt = build_sentiment_prompt(
            df.drop(i), "sad", examples_per_class, same_name
        )

        prompt = happy_prompt + sad_prompt
        random.shuffle(prompt)

        prompt = prompt + [query]

        inputs.append("\n".join(prompt))
        targets.append(target)

    return inputs, targets


def plot_from_jsonl(file_path):
    # Defining line style, marker and color arrays to ensure distinct appearance for each model's plot
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    with jsonlines.open(file_path, mode="r") as reader:
        for idx, line in enumerate(reader):
            model_name = line["model"]
            ex_class = line["ex_class"]
            avg_accs = line["avg_accs"]

            # Assigning line style, marker, and color for each model's plot from predefined arrays
            line_style = line_styles[idx % len(line_styles)]
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]

            # plot
            plt.plot(
                ex_class,
                avg_accs,
                label=model_name,
                linestyle=line_style,
                marker=marker,
                color=color,
                alpha=0.7,
            )

    plt.xlabel("Examples per class")
    plt.ylabel("Accuracy")
    plt.title("Few-shot sentiment classification")
    plt.legend(prop={'size': 8})
    plt.show()


def main():
    n_seeds = 5
    examples_per_class = range(10)
    model_names = [
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "roneneldan/TinyStories-1M",
        "roneneldan/TinyStories-3M",
        "roneneldan/TinyStories-8M",
        "roneneldan/TinyStories-28M",
        "roneneldan/TinyStories-33M",
    ]

    # Defining line style, marker and color arrays to ensure distinct appearance for each model's plot
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for idx, model_name in enumerate(model_names):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        avg_accs = []
        ex_class = []

        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        print(f"Initialized {model_name} with {n_params/2**20:.2f}M params")
        model.to("cuda")

        seeds = range(n_seeds)
        for n in examples_per_class:
            accs = []
            for seed in seeds:
                set_seed(seed)
                inputs, targets = build_sentiment_dataset(examples_per_class=n)
                correct = []
                preds = []
                for i, (input, target) in enumerate(zip(inputs, targets)):
                    input_ids = tokenizer.encode(input, return_tensors="pt").to("cuda")
                    output = model.generate(
                        input_ids,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    pred = output_text.split()[-1]
                    preds.append(pred)
                    if pred == target:
                        correct.append(1)
                    else:
                        correct.append(0)
                accs.append(sum(correct) / len(correct))
            avg_acc = sum(accs) / len(accs)
            print(f"Examples per class: {n}, Accuracy: {avg_acc}")
            avg_accs.append(avg_acc)
            ex_class.append(n)

        # Assigning line style, marker, and color for each model's plot from predefined arrays
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]

        # plot
        plt.plot(
            ex_class,
            avg_accs,
            label=model_name,
            linestyle=line_style,
            marker=marker,
            color=color,
        )

        # save avg_accs and ex_class as jsonlines and append to file
        line = {"model": model_name, "ex_class": ex_class, "avg_accs": avg_accs}

        # append to file
        with jsonlines.open("acc_vs_ex.jsonl", mode="a") as writer:
            writer.write(line)

    plt.xlabel("Examples per class")
    plt.ylabel("Accuracy")
    plt.title("Few-shot sentiment classification")
    plt.legend()
    # save
    plt.savefig("acc_vs_ex.png")


if __name__ == "__main__":
    # main()
    plot_from_jsonl("acc_vs_ex.jsonl")
