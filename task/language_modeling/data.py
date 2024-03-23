"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

import numpy as np

# import requests
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import einops
from einops import rearrange, reduce

import transformers
import datasets

IGNORE_IDX = -1
# from tqdm import tqdm

# from tokenizer import Tokenizer

# # TODO make this configurable
# DATA_CACHE_DIR = "/graft1/datasets/shared"


# def download_file(url: str, fname: str, chunk_size=1024):
#     """Helper function to download a file from a given url"""
#     resp = requests.get(url, stream=True)
#     total = int(resp.headers.get("content-length", 0))
#     with open(fname, "wb") as file, tqdm(
#         desc=fname,
#         total=total,
#         unit="iB",
#         unit_scale=True,
#         unit_divisor=1024,
#     ) as bar:
#         for data in resp.iter_content(chunk_size=chunk_size):
#             size = file.write(data)
#             bar.update(size)


# def download():
#     """Downloads the dataset to disk."""
#     os.makedirs(DATA_CACHE_DIR, exist_ok=True)

#     # download the TinyStories dataset, unless it's already downloaded
#     data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
#     data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
#     if not os.path.exists(data_filename):
#         print(f"Downloading {data_url} to {data_filename}...")
#         download_file(data_url, data_filename)
#     else:
#         print(f"{data_filename} already exists, skipping download...")

#     # unpack the tar.gz file into all the data shards (json files)
#     data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir, exist_ok=True)
#         print(f"Unpacking {data_filename}...")
#         os.system(f"tar -xzf {data_filename} -C {data_dir}")
#     else:
#         print(f"{data_dir} already exists, skipping unpacking...")

#     # print a single example just for debugging and such
#     shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
#     with open(shard_filenames[0], "r") as f:
#         data = json.load(f)
#     print("Download done.")
#     print(f"Number of shards: {len(shard_filenames)}")
#     print(f"Example story:\n{data[0]}")


# def pretokenize():
#     enc = Tokenizer()

#     def process_shard(shard):
#         with open(shard, "r") as f:
#             data = json.load(f)
#         all_tokens = []
#         for example in tqdm(data):
#             text = example["story"]
#             text = text.strip()  # get rid of leading/trailing whitespace
#             tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
#             all_tokens.extend(tokens)
#         # convert to uint16 nparray
#         all_tokens = np.array(all_tokens, dtype=np.uint16)
#         # write to disk
#         tokenized_filename = shard.replace(".json", ".bin")
#         with open(tokenized_filename, "wb") as f:
#             f.write(all_tokens.tobytes())
#         print(f"Saved {tokenized_filename}")

#     # iterate the shards and tokenize all of them one by one
#     data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
#     shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

#     # process all the shards in a threadpool
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         executor.map(process_shard, shard_filenames)

#     print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, data_dir, generators, seed):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir
        self.seed = seed
        self.generators = generators

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def __iter__(self):
        # get worker info within a DataLoader
        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        # seed = self.seed + worker_id + 1337 * rank
        # rng = random.Random(seed)
        # print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(self.data_dir, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = (
        #     shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        # )
        if self.split == "train":
            shard_filenames = shard_filenames[2:]
        elif self.split == "eval":
            shard_filenames = shard_filenames[0:1]
        elif self.split == "test":
            shard_filenames = shard_filenames[1:2]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        while True:
            self.generators["random"].shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                self.generators["random"].shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield {"inputs": x, "targets": y}


class Task:
    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


class LanguageModelEmbedder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = torch.nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)

    def forward(self, batch: dict[str, torch.Tensor]):
        inputs = batch["inputs"]
        return self.embedder(inputs)


class LanguageModelHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.head = torch.nn.Linear(cfg.model.d_model, cfg.data.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.head(x)


def lm_loss_fn(logits, targets, batch):
    cel = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX, reduction="none")
    batch_size = logits.shape[0]
    logits = rearrange(logits, "b s c -> (b s) c")
    targets = rearrange(targets, "b s -> (b s)")
    loss = cel(logits, targets)
    loss = rearrange(loss, "(b s) -> b s", b=batch_size)
    loss = reduce(loss, "b s -> s", reduction="mean")
    return loss


def lm_hf_loss_fn(logits, targets, batch, _=None):
    cel = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX, reduction="none")
    batch_size = logits.shape[0]
    logits = rearrange(logits, "b s c -> (b s) c")
    targets = rearrange(targets, "b s -> (b s)")
    loss = cel(logits, targets)
    loss = rearrange(loss, "(b s) -> b s", b=batch_size)
    return loss


@torch.inference_mode()
def generate(
    model,
    embedder,
    head,
    batch,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    max_seq_len=128,
    prompt_len=50,
):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    Also note this is a super inefficient version of sampling with no key/value cache.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        # idx_cond = idx if (idx.size(1) <= max_seq_len) else idx[:, -max_seq_len:]
        # forward the model to get the logits for the index in the sequence
        embeddings = embedder(batch)  # [b,t,d]
        embeddings = (
            embeddings
            if (embeddings.size(1) <= max_seq_len)
            else embeddings[:, -max_seq_len:, :]
        )
        hidden_states = model(embeddings)  # [b,t,d]
        logits = head(hidden_states)  # [b,t,v]
        logits = logits[:, -1, :]  # crop to just the final time step
        if temperature == 0.0:
            # "sample" the single most likely index
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        # idx = torch.cat((idx, idx_next), dim=1)
        batch["inputs"] = torch.cat((batch["inputs"], idx_next), dim=1)

    return batch["inputs"]


def get_training_corpus(raw_datasets, split="train"):
    return (
        raw_datasets[split][i : i + 1000]["text"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


def train_tokenizer(raw_datasets, cfg):
    training_corpus = get_training_corpus(raw_datasets, "train")
    old_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf"
    )
    length = len(raw_datasets["train"]) // 1000
    tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus, vocab_size=cfg.data.vocab_size, length=length
    )
    tokenizer.save_pretrained(cfg.data.tokenizer_dir)
    return tokenizer


def build_lm_dataset(cfg):
    # 5. Load and preprocess dataset
    raw_datasets = datasets.load_dataset("roneneldan/TinyStories")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.data.tokenizer_dir)
    except Exception as e:
        print(f"Failed to load tokenizer from {cfg.data.tokenizer_dir}. Error: {e}")
        # print current working directory
        print(f"currently in {os.getcwd()}")
        print("Training new tokenizer from scratch...")
        tokenizer = train_tokenizer(raw_datasets, cfg)

    # tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_eos_token = True
    cfg.data.vocab_size = tokenizer.vocab_size
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text"

    # 5.1. Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.data.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not cfg.data.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # 5.2 Concatenate all texts and turn them into blocks for batching
    block_size = cfg.data.max_seq_len

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=cfg.data.preprocessing_num_workers,
        load_from_cache_file=not cfg.data.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    # convert all tokens to torch tensors
    lm_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    lm_datasets = lm_datasets.rename_column("input_ids", "inputs")
    lm_datasets = lm_datasets.rename_column("labels", "targets")
    return lm_datasets


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_function(examples, tokenizer, text_column_name):
    return tokenizer(examples[text_column_name])


class HFLanguageModelDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, generators, seed, cfg):
        super().__init__()

        self.split = split
        self.generators = generators
        self.seed = seed
        self.block_size = cfg.data.max_seq_len + 1

        if cfg.data.version == "original":
            data_files = None
        elif cfg.data.version == "gpt4_only":
            data_files = {
                "train": "TinyStoriesV2-GPT4-train.txt",
                "validation": "TinyStoriesV2-GPT4-valid.txt",
            }
        elif cfg.data.version == "union":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported data version: {cfg.data.version}")

        raw_datasets = datasets.load_dataset(
            "roneneldan/TinyStories",
            data_files=data_files,
        )
        tokenizer = self.load_tokenizer(raw_datasets, cfg)
        self.tokenizer = tokenizer

        tokenizer.add_eos_token = True
        cfg.data.vocab_size = tokenizer.vocab_size
        self.cfg = cfg

        column_names = list(raw_datasets["train"].features)
        text_column_name = "text"

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Running tokenizer on dataset",
            fn_kwargs={"tokenizer": tokenizer, "text_column_name": text_column_name},
        )
        if split == "train":
            self.tokenized_datasets = tokenized_datasets["train"]
        elif split == "eval":
            self.tokenized_datasets = tokenized_datasets["validation"]
        else:
            raise ValueError(f"Unsupported split: {split}")

    def shuffle(self):
        self.tokenized_datasets = self.tokenized_datasets.shuffle(
            generator=self.generators["numpy"]
        )
        print("Shuffled dataset.")

    def group_texts(self):
        grouped_texts = self.tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.cfg.data.preprocessing_num_workers,
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {self.block_size}",
            fn_kwargs={"block_size": self.block_size},
        )
        grouped_texts.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        return grouped_texts

    def load_tokenizer(self, raw_datasets, cfg):
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                cfg.data.tokenizer_dir
            )
        except Exception as e:
            print(f"Failed to load tokenizer from {cfg.data.tokenizer_dir}. Error: {e}")
            print("Training new tokenizer from scratch.")
            tokenizer = train_tokenizer(raw_datasets, cfg)

        return tokenizer

    def reset(self):
        self.generators["torch"].manual_seed(self.seed)
        self.generators["numpy"] = np.random.default_rng(self.seed)
        self.generators["random"] = random.Random(self.seed)

    def __iter__(self):
        while True:
            if self.split == "train" and self.cfg.data.do_shuffle:
                self.shuffle()
            grouped_text = self.group_texts()
            for example in grouped_text:
                yield {
                    "inputs": example["input_ids"][:-1],
                    "targets": example["labels"][1:],
                }
