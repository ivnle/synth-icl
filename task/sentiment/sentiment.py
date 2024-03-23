import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import transformers
import numpy as np


class SentimentDataset(Dataset):
    def __init__(self, generators, seed, cfg):
        self.do_flip_class = cfg.data.sent.do_flip_class
        self.generators = generators
        self.filepath = cfg.data.sent.filepath
        self.examples_per_class = cfg.data.sent.examples_per_class
        self.same_name = cfg.data.sent.same_name
        self.inputs, self.targets = self.build_sentiment_dataset()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.data.tokenizer_dir
        )
        self.seed = seed

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def build_sentiment_prompt(self, df, emotion, n_examples, same_name):
        _df = df[df["Class"] == emotion]
        rows = _df.sample(
            (n_examples), replace=False, random_state=self.generators["numpy"]
        )
        samples = []
        for row in rows.iterrows():
            sample, _ = self.build_sentiment_sample(row[1], same_name)
            samples.append(sample)
        return samples

    def build_sentiment_sample(self, row, same_name=None, is_query=False):
        subject = row["Subject"]
        if same_name is not None:
            subject = same_name
        sentence = row["Sentence"].replace(row["Subject"], subject)
        target = row["Class"]

        if self.do_flip_class:
            if target == self.classes[0]:
                target = self.classes[1]
            elif target == self.classes[1]:
                target = self.classes[0]
            else:
                raise ValueError(f"Invalid class `{target}`")

        if is_query:
            input = f"{sentence} {subject} is"
        else:
            input = f"{sentence} {subject} is {target}."
        return input, target

    def build_sentiment_dataset(self):
        df = pd.read_csv(self.filepath)

        # get unique Class values and save as tuple
        self.classes = tuple(df["Class"].unique())

        inputs, targets = [], []
        for i, row in enumerate(df.iterrows()):
            query, target = self.build_sentiment_sample(
                row[1], self.same_name, is_query=True
            )
            happy_prompt = self.build_sentiment_prompt(
                df.drop(i), "happy", self.examples_per_class, self.same_name
            )
            sad_prompt = self.build_sentiment_prompt(
                df.drop(i), "sad", self.examples_per_class, self.same_name
            )

            prompt = happy_prompt + sad_prompt
            random.shuffle(prompt)

            prompt = prompt + [query]

            inputs.append("\n".join(prompt))
            targets.append(target)
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        tokenized_input = self.tokenizer(input, return_tensors="pt")["input_ids"]
        tokenized_input = tokenized_input.squeeze(0)
        return {
            "inputs": tokenized_input,
            "targets": tokenized_input,
            "dec_inputs": input,
            "dec_targets": target,
        }
