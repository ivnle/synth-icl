import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision
import transformers
from transformers.utils import logging
import datasets
import pandas as pd
import argparse
import copy
import json
import jsonlines

import math
from collections import defaultdict
import numpy as np
import os
from pathlib import Path
import random
import wandb
import hydra
from contextlib import nullcontext
from omegaconf import DictConfig, OmegaConf
import arch.safari.utils as utils

# lucidrains implementation
# import sinkhorn_transformer

from tqdm import tqdm
import einops
from einops import rearrange, reduce

logger = logging.get_logger("transformers")

# import tasks
from task import (
    LanguageModelEmbedder,
    LanguageModelDataset,
    LanguageModelHead,
    HFLanguageModelDataset,
    build_lm_dataset,
    lm_loss_fn,
    generate,
    LanguageModelTokenizer,
    LRDataset,
    LRHead,
    LREmbedder,
    lr_loss_fn,
    lm_hf_loss_fn,
    lr_loss_fn_parallel,
    GMMDataset,
    MCCDataset,
    MCCEmbedder,
    MCCHead,
    mcc_loss_fn,
    mcc_loss_fn_parallel,
    ARDataset,
    ARDatasetSafari,
    AREmbedder,
    ARHead,
    ar_loss_fn,
    ar_loss_fn_parallel,
    OmniglotDataset,
    OmniglotEmbedder,
    OmniglotHead,
    SeqGenerator,
    OmniglotDatasetForSampling,
    omniglot_loss_fn,
    SentimentDataset,
)

# import models
from arch import (
    HFModel,
    LlamaModel,
    LSTMModel,
    RetNetConfig,
    RetNetModel,
    LightConvConfig,
    LightConvModel,
    GRUModel,
    RNNModel,
    HyperMixerModel,
    XTransformersModel,
    SimpleLMHeadModel,
    MambaModel,
)
from mamba_ssm.models.config_mamba import MambaConfig

# allows for interpolation with operators in hydra yamls
OmegaConf.register_new_resolver("eval", eval)

# go fast
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# tasks
AR = "assoc-recall"
LM = "language-modeling"
LM_HF = "language-modeling-HF"
LR = "linear-regression"
MCC = "multiclass-classification"
GMM = "gauss-mix-model"
OG = "omniglot"
SENT = "sentiment"

# architectures
T5 = "t5"
SAFARI = "safari"
MAMBA = "mamba"
RWKV = "rwkv"
RWKV_HF = "rwkv_hf"  # this results in nan loss with ninja
RETNET = "retnet"
TRAN = "transformer"
RNN = "rnn"
LSTM = "lstm"
GRU = "gru"
REFORMER = "reformer"  # https://arxiv.org/abs/2001.04451
SINKHORN = "sinkhorn"
FNET = "fnet"  # https://arxiv.org/abs/2105.03824
GPT2 = "gpt2"
BERT = "bert"
ROBERTA = "roberta"
LIGHT_CONV = "lightconv"
DYNAMIC_CONV = "dynamicconv"
HYPERMIXER = "hypermixer"
GPT2X = "gpt2-x"
LlamaX = "llama-x"
BERTX = "bert-x"
T5X = "t5-x"
X_DEC = "x-decoder"
X_ENC = "x-encoder"
X_ENC_DEC = "x-encoder-decoder"

TRAIN = "train"
EVAL = "eval"
TEST = "test"
EVAL_TRAIN = "eval_train"
EVAL_EVAL = "eval_eval"
EVAL_TEST = "eval_test"

IGNORE_IDX = -1

head_types = {
    AR: ARHead,
    LM: LanguageModelHead,
    LM_HF: LanguageModelHead,
    LR: LRHead,
    MCC: MCCHead,
    GMM: MCCHead,
    OG: OmniglotHead,
}

embedder_types = {
    AR: AREmbedder,
    LM: LanguageModelEmbedder,
    LM_HF: LanguageModelEmbedder,
    LR: LREmbedder,
    MCC: MCCEmbedder,
    GMM: MCCEmbedder,
    OG: OmniglotEmbedder,
}

loss_fn_types = {
    AR: ar_loss_fn,
    LM: lm_loss_fn,
    LM_HF: lm_hf_loss_fn,
    LR: lr_loss_fn,
    MCC: mcc_loss_fn,
    GMM: mcc_loss_fn,
    OG: omniglot_loss_fn,
}

loss_fn_types_parallel = {
    AR: ar_loss_fn_parallel,
    MCC: mcc_loss_fn_parallel,
    GMM: mcc_loss_fn_parallel,
    LR: lr_loss_fn_parallel,
    LM: lm_loss_fn,
    LM_HF: lm_hf_loss_fn,
    OG: omniglot_loss_fn,
}


class ModelWrapper(nn.Module):
    def __init__(
        self,
        embedder,
        model,
        head,
    ):
        super().__init__()

        self.embedder = embedder
        self.model = model
        self.head = head

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    n_params = 0
    ignore_names = ["head.fc", "head.head", "embedder.embedder", "wte", "wpe", "embed"]
    ignored = []
    for name, param in model.named_parameters():
        for ignore_name in ignore_names:
            # continue to the next named param
            if ignore_name in name:
                ignored.append((name, param.numel()))
                break
        else:
            # if we didn't break out of the loop, then we didn't encounter an ignore_name
            if param.requires_grad:
                print(f"{name}: {param.numel()}")
                n_params += param.numel()
        wandb.run.summary.update(
            {
                "n_params": n_params,
            }
        )
    print(f"{n_params=}")
    print(f"{ignored=}")


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


def add_example_to_table(
    batch: dict, tables, split: str, in_eval: bool, train_iter, cfg, eval_iter=None
):
    for batch_idx in cfg.log_batch_idx:
        # pick random idx from batch
        if cfg.data._name_ == OG:
            batch_size = batch["example"].shape[0]
        else:
            batch_size = batch["inputs"].shape[0]
        if batch_idx >= batch_size:
            continue
        
        input = batch["inputs"][batch_idx].tolist()
        target = batch["targets"][batch_idx].tolist()
        if in_eval:
            # table[split].add_data(train_iter, eval_iter, batch_idx, input, target)
            print(f"[[split={split}]] train_iter: {train_iter}, eval_iter: {eval_iter}, batch_idx: {batch_idx}, input: {input[:10]}, target: {target[:10]}")
        else:
            # tables[TRAIN].add_data(train_iter, batch_idx, input, target)
            print(f"[[split=train]] train_iter: {train_iter}, batch_idx: {batch_idx}, input: {input[:10]}, target: {target[:10]}")

    return
    table = {
        TRAIN: tables[EVAL_TRAIN],
        EVAL: tables[EVAL_EVAL],
        TEST: tables[EVAL_TEST],
    }
    for batch_idx in cfg.log_batch_idx:
        # pick random idx from batch
        if cfg.data._name_ == OG:
            batch_size = batch["example"].shape[0]
        else:
            batch_size = batch["inputs"].shape[0]
        if batch_idx >= batch_size:
            continue

        # log examples to sanity check determinism
        if cfg.data._name_ == LM:
            tokenizer = LanguageModelTokenizer()
            to_deocde = batch["inputs"][batch_idx].tolist()
            decoded = tokenizer.decode(to_deocde)
            # print(f"train_iter: {train_iter}, train_input: {repr(decoded)}")

            # get first 100 and last 100 elements of decoded
            trunc_decoded = decoded[:100] + "..." + decoded[-100:]
            if in_eval:
                table[split].add_data(train_iter, eval_iter, batch_idx, trunc_decoded)
            else:
                tables[TRAIN].add_data(train_iter, batch_idx, trunc_decoded)
        elif cfg.data._name_ == OG:
            input = batch["example"][batch_idx]  # [9, 105, 105, 1]
            input = rearrange(input, "b h w c -> b c h w")
            input = torchvision.utils.make_grid(input, nrow=9)
            target = batch["target"][batch_idx]
            # ignore odd indices
            target = target[::2]
            if in_eval:
                table[split].add_data(
                    train_iter, eval_iter, batch_idx, wandb.Image(input), target
                )
            else:
                tables[TRAIN].add_data(
                    train_iter, batch_idx, wandb.Image(input), target
                )

        else:
            input = batch["inputs"][batch_idx].tolist()
            target = batch["targets"][batch_idx].tolist()
            if in_eval:
                table[split].add_data(train_iter, eval_iter, batch_idx, input, target)
            else:
                tables[TRAIN].add_data(train_iter, batch_idx, input, target)


def build_dataloader(dataset, collate_fn, split, cfg, seed_offset=0):
    batch_size = cfg.train.batch_size if split == TRAIN else cfg.eval.batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        worker_init_fn=seed_worker,
        # generator=g,
        # shuffle=do_shuffle,
    )

    return dataloader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def running_average(tensor):
    cumulative_sum = tensor.cumsum(dim=-1)
    index = torch.arange(
        1, tensor.size(-1) + 1, dtype=torch.float, device=tensor.device
    )
    return cumulative_sum / index


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, cfg):
    warmup_iters = cfg.scheduler.warmup_iters
    lr_decay_iters = cfg.scheduler.lr_decay_iters
    learning_rate = cfg.optimizer.lr
    min_lr = cfg.scheduler.min_lr
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def build_head(cfg) -> nn.Module:
    """Returns an nn.Module that maps final hidden states to logits."""
    dataset = cfg.data._name_
    if dataset not in head_types:
        raise ValueError(f"Unknown dataset: {dataset}")

    # hacky way to handle reformer doubling d_model
    if cfg.model._name_ == REFORMER:
        orig_d_model = cfg.model.d_model
        cfg.model.d_model = orig_d_model * 2
        head = head_types[dataset](cfg)
        cfg.model.d_model = orig_d_model
    else:
        head = head_types[dataset](cfg)

    return head


def build_embedder(cfg) -> nn.Module:
    """Returns a nn.Module that maps sequences of inputs to a sequences of embeddings.
    Example: an embedding layer that maps integer tokens to embeddings."""
    dataset = cfg.data._name_
    if dataset not in embedder_types:
        raise ValueError(f"Unknown dataset: {dataset}")
    return embedder_types[dataset](cfg)


def build_model(cfg, embedder, head) -> nn.Module:
    """Returns a nn.Module that maps embeddings to final hidden states."""
    model_name = cfg.model._name_
    if model_name == TRAN:
        # Andrej Karpathy's implementation of Llama2
        model = LlamaModel(params=cfg.model, embedder=embedder, head=head)
    elif model_name == "mega":
        config = transformers.MegaConfig(
            vocab_size=1,
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layer,
            intermediate_size=cfg.model.d_inner,
            bidirectional=False,
            max_positions=cfg.model.max_seq_len,
            is_decoder=True,
            pad_token_id=0,  # HF will complain if we don't override
        )
        model = transformers.MegaModel(config)
        model = HFModel(model, embedder, head, cfg)
    # elif model_name == SINKHORN:
    #     model = sinkhorn_transformer.SinkhornTransformer(
    #         causal=True,
    #         dim=cfg.model.d_model,
    #         heads=int(cfg.model.n_heads),
    #         depth=cfg.model.n_layer,
    #         bucket_size=cfg.model.bucket_size,
    #     )
    #     model = ModelWrapper(embedder, model, head)
    elif model_name == REFORMER:
        # set number of layers
        assert cfg.model.n_layer % 2 == 0
        attn_layers = ["local", "lsh"] * (cfg.model.n_layer // 2)

        config = transformers.ReformerConfig(
            hidden_size=cfg.model.d_model,
            attn_layers=attn_layers,
            feed_forward_size=cfg.model.d_inner,
            is_decoder=True,
            axial_pos_embds=False,
            num_attention_heads=int(cfg.model.n_heads),
            vocab_size=1,
            pad_token_id=0,
        )
        model = transformers.ReformerModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == "llama2":
        # https://huggingface.co/docs/transformers/model_doc/llama2
        config = transformers.LlamaConfig(
            vocab_size=1,  # HF will init its own embed, 1 makes it small
            hidden_size=cfg.model.d_model,
            intermediate_size=cfg.model.d_inner,
            num_hidden_layers=cfg.model.n_layer,
            num_attention_heads=int(cfg.model.n_heads),
            num_key_value_heads=int(cfg.model.n_kv_heads),
            # pretraining_tp=,
            # hidden_act=,
            max_position_embeddings=cfg.model.max_seq_len,
            # initializer_range=,
            # rms_norm_eps=,
            # use_cache=,
            # tie_word_embeddings=,
            # rope_scaling=,
        )
        model = transformers.LlamaModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == LlamaX:
        model = XTransformersModel(cfg, embedder, head)
    elif model_name == T5:
        if cfg.model.n_layer % 2 != 0:
            raise ValueError("n_layer must be even for T5")
        config = transformers.T5Config(
            vocab_size=1,  # just pass one learnable embedding
            d_model=cfg.model.d_model,
            d_kv=int(cfg.model.d_model / cfg.model.n_heads),
            d_ff=cfg.model.d_inner,
            num_layers=int(cfg.model.n_layer / 2),
            num_decoder_layers=int(cfg.model.n_layer_decoder / 2),
            num_heads=int(cfg.model.n_heads),
            dropout_rate=cfg.model.dropout,
            decoder_start_token_id=0,
        )
        model = transformers.T5Model(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == T5X:
        model = XTransformersModel(cfg, embedder, head)
    elif model_name == LIGHT_CONV:
        config = LightConvConfig(
            encoder_conv_type=cfg.model.conv_type,
            decoder_conv_type=cfg.model.conv_type,
            weight_softmax=cfg.model.weight_softmax,
            encoder_embed_dim=cfg.model.d_model,
            decoder_embed_dim=cfg.model.d_model,
            encoder_ffn_embed_dim=cfg.model.d_inner,
            decoder_ffn_embed_dim=cfg.model.d_inner,
            encoder_layers=cfg.model.n_encoding_layer,  # TODO: encoder_layers != decoder_layers and has special setup, change in a bit
            decoder_layers=cfg.model.n_decoding_layer,
            encoder_attention_heads=int(
                cfg.model.n_heads
            ),  # TODO: same for above reason
            decoder_attention_heads=int(cfg.model.n_heads),
            max_source_positions=cfg.model.max_seq_len,  # TODO: should we match max_source_positions and max_target_positions?
            max_target_positions=cfg.model.max_seq_len,
            decoder_only=cfg.model.decoder_only,
        )
        model = LightConvModel(config, embedder, head)
    elif model_name == DYNAMIC_CONV:
        config = LightConvConfig(
            encoder_conv_type=cfg.model.conv_type,
            decoder_conv_type=cfg.model.conv_type,
            weight_softmax=cfg.model.weight_softmax,
            encoder_embed_dim=cfg.model.d_model,
            decoder_embed_dim=cfg.model.d_model,
            encoder_ffn_embed_dim=cfg.model.d_inner,
            decoder_ffn_embed_dim=cfg.model.d_inner,
            encoder_layers=cfg.model.n_encoding_layer,  # TODO: encoder_layers != decoder_layers and has special setup, change in a bit
            decoder_layers=cfg.model.n_decoding_layer,
            encoder_attention_heads=int(cfg.model.n_heads),
            decoder_attention_heads=int(cfg.model.n_heads),
            max_source_positions=cfg.model.max_seq_len,
            max_target_positions=cfg.model.max_seq_len,
            decoder_only=cfg.model.decoder_only,
        )
        model = LightConvModel(config, embedder, head)
    elif model_name == BERT:
        config = transformers.BertConfig(
            vocab_size=1,
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layer,
            num_attention_heads=int(cfg.model.n_heads),
            intermediate_size=cfg.model.d_inner,
            # hidden_act='gelu',
            hidden_dropout_prob=cfg.model.dropout,
            attention_probs_dropout_prob=cfg.model.dropout,
            max_position_embeddings=cfg.model.max_seq_len,
            # type_vocab_size=2,
            # initializer_range=0.02,
            # layer_norm_eps=1e-12,
            # position_embedding_type='absolute',
        )
        model = transformers.BertModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == BERTX:
        model = XTransformersModel(cfg, embedder, head)
    elif model_name == ROBERTA:
        config = transformers.RobertaConfig(
            vocab_size=1,
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layer,
            num_attention_heads=int(cfg.model.n_heads),
            intermediate_size=cfg.model.d_inner,
            # hidden_act='gelu',
            hidden_dropout_prob=cfg.model.dropout,
            attention_probs_dropout_prob=cfg.model.dropout,
            max_position_embeddings=cfg.model.max_seq_len,
            # type_vocab_size=2,
            # initializer_range=0.02,
            # layer_norm_eps=1e-12,
            # position_embedding_type='absolute',
            pad_token_id=0,  # HF will complain if we don't override
        )
        model = transformers.RobertaModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == HYPERMIXER:
        model = HyperMixerModel(cfg.model, embedder, head)
    elif model_name == LSTM:
        model = LSTMModel(
            embedder=embedder,
            head=head,
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=cfg.model.n_layer,
            bias=cfg.model.bias,
            dropout=cfg.model.dropout,
            # feed_forward_proj=cfg.model.activ_fn,
        )
    elif model_name == RNN:
        model = RNNModel(
            embedder=embedder,
            head=head,
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=cfg.model.n_layer,
            nonlinearity=cfg.model.activ_fn,
            bias=cfg.model.bias,
            batch_first=cfg.model.batch_first,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
        )
    elif model_name == GRU:
        model = GRUModel(
            embedder=embedder,
            head=head,
            input_size=cfg.model.d_model,
            hidden_size=cfg.model.d_model,
            num_layers=cfg.model.n_layer,
            bias=cfg.model.bias,
            batch_first=cfg.model.batch_first,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
        )
    elif model_name == SAFARI:
        model = SimpleLMHeadModel(**cfg.model, embedder=embedder, head=head)
    elif model_name == MAMBA:
        config = MambaConfig(
            d_model=cfg.model.d_model,
            n_layer=cfg.model.n_layer,            
        )
        model = MambaModel(config, embedder, head)
    elif model_name == RWKV:
        # set env vars to make rwkv happy
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_T_MAX"] = str(cfg.model.max_seq_len)

        dtype2prec = {
            "float32": "fp32",
            "tf32": "tf32",
            "float16": "fp16",
            "bfloat16": "bf16",
        }
        os.environ["RWKV_FLOAT_MODE"] = dtype2prec[cfg.train.dtype]

        from arch.rwkv.model import RWKVModel

        config = {
            "ctx_len": cfg.model.max_seq_len,
            "n_layer": cfg.model.n_layer,
            "n_embd": cfg.model.d_model,
            "my_pos_emb": 0,
            "pre_ffn": 0,
            "dropout": 0,
            "head_qk": 0,
            "grad_cp": 0,
        }
        config = argparse.Namespace(**config)
        model = RWKVModel(config, embedder, head)

    elif model_name == RWKV_HF:
        # https://huggingface.co/docs/transformers/model_doc/rwkv
        config = transformers.RwkvConfig(
            # vocab_size=cfg.data.vocab_size,
            context_length=cfg.model.max_seq_len,
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layer,
            # attention_hidden_size=cfg.model.attention_hidden_size,
            intermediate_size=cfg.model.d_inner,
            # layer_norm_eps=cfg.model.norm_eps,
            # rescale_every=cfg.model.rescale_every,
        )
        model = transformers.RwkvModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == RETNET:
        config = RetNetConfig(
            decoder_embed_dim=cfg.model.d_model,
            decoder_retention_heads=int(cfg.model.n_heads),
            decoder_ffn_embed_dim=cfg.model.d_inner,
            decoder_layers=cfg.model.n_layer,
            max_target_positions=cfg.model.max_seq_len,
            # moe_gating_use_fp32=cfg.train.dtype == "float32", # TODO should we add this?
        )
        model = RetNetModel(config, embedder, head)
    elif model_name == FNET:
        # https://huggingface.co/docs/transformers/model_doc/fnet
        # sequence length must be a power of 2 for cuFFT to work
        config = transformers.FNetConfig(
            vocab_size=1,  # HF will init its own embed, 1 makes it small
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layer,
            intermediate_size=cfg.model.d_inner,
            # hidden_act="gelu_new",  # TODO make this a config
            hidden_dropout_prob=cfg.model.dropout,
            max_position_embeddings=cfg.model.max_seq_len,
            # type_vocab_size,
            # initializer_range,
            # layer_norm_eps,
            # use_tpu_fourier_optimizations,
            # tpu_short_seq_length,
            pad_token_id=0,  # HF will complain if we don't override
        )
        model = transformers.FNetModel(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == GPT2:
        # https://huggingface.co/docs/transformers/model_doc/gpt2
        config = transformers.GPT2Config(
            vocab_size=1,
            n_positions=cfg.model.max_seq_len,
            n_embd=cfg.model.d_model,
            n_layer=cfg.model.n_layer,
            n_head=int(cfg.model.n_heads),
            n_inner=cfg.model.d_inner,
            # activation_function="gelu_new", # TODO make this a config
            resid_pdrop=cfg.model.dropout,
            embd_pdrop=cfg.model.dropout,
            attn_pdrop=cfg.model.dropout,
            # layer_norm_epsilon=cfg.model["norm_eps"],
        )
        model = transformers.GPT2Model(config)
        model = HFModel(model, embedder, head, cfg)
    elif model_name == GPT2X:
        model = XTransformersModel(cfg, embedder, head)
    elif model_name in [X_DEC, X_ENC, X_ENC_DEC]:
        model = XTransformersModel(cfg, embedder, head)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # TODO resuming training needs to also account for all parameters used to train the model
    # i.e. we need to capture and load `cfg`. Also need to account for what training step we
    # left off on, learning rate, etc. Look up best practices.
    if "resume_path" in cfg.model:
        model.load_state_dict(
            torch.load(cfg.model.resume_path, map_location=cfg.device)
        )
    # compile the model
    # leads to unstable eval loss curve. why? leave false for now
    if cfg.train.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    if cfg.data._name_ == LM:
        # share the unembedding parameters with the embedding parameters
        model.embedder.embedder.weight = model.head.head.weight

    return model


def build_loss_fn(cfg, in_eval: bool) -> callable:
    """
    Returns a callable with the following signature
    inputs:
        logits: tensor of shape [b, s, c]
        targets: tensor of shape [b, s]
        batch: dict
    output:
        loss: tensor of shape [s]
    """
    dataset = cfg.data._name_
    if dataset not in loss_fn_types:
        raise ValueError(f"Unknown dataset: {dataset}")
    if cfg.train.parallel_loss or in_eval:
        return loss_fn_types_parallel[dataset]
    else:
        return loss_fn_types[dataset]


def build_datasets(generators, cfg) -> tuple[Dataset, Dataset]:
    """Returns the two pytorch datasets used to train and evaluate the model."""
    dataset = cfg.data._name_
    if dataset == OG:
        dataset_for_sampling = OmniglotDatasetForSampling(**cfg.data.omniglot_config)

        data_generator_factory = SeqGenerator(
            dataset_for_sampling=dataset_for_sampling,
            random_seed=cfg.seed,
            **cfg.data.generator_config,
        )
        cfg.data.num_classes = data_generator_factory.n_classes

        # train dataset
        if cfg.data.train_seqs == "bursty":
            seq_generator = data_generator_factory.get_bursty_seq
            generator_args = cfg.data.seq_config
        else:
            raise ValueError
        # examples of the form [example, label, is_rare, targets]
        train_dataset = OmniglotDataset(
            seq_generator, generator_args, generators, cfg.seed
        )

        # eval dataset
        if cfg.data.eval_seqs == "fewshot_holdout":
            seq_generator = data_generator_factory.get_fewshot_seq
            generator_args = {
                "class_type": "holdout",
                "shots": cfg.data.seq_config.fs_shots,
                "ways": cfg.data.seq_config.ways,
                "labeling": "unfixed",
                "randomly_generate_rare": cfg.data.seq_config.randomly_generate_rare,
                "grouped": cfg.data.seq_config.grouped,
            }
        elif cfg.data.train_seqs == "no_support_zipfian":
            raise NotImplementedError
            seq_generator = data_generator_factory.get_no_support_seq
            all_unique = False
            generator_args = (
                "zipfian",
                cfg.seq_len,
                all_unique,
                cfg.labeling_common,
                cfg.randomly_generate_rare,
            )
        else:
            raise ValueError
        eval_dataset = OmniglotDataset(
            seq_generator, generator_args, generators, cfg.seed_eval
        )
        test_dataset = OmniglotDataset(
            seq_generator, generator_args, generators, cfg.seed_test
        )
    elif dataset == LM_HF:
        train_dataset = HFLanguageModelDataset(TRAIN, generators[TRAIN], cfg.seed, cfg)
        eval_dataset = HFLanguageModelDataset(
            EVAL, generators[EVAL], cfg.seed_eval, cfg
        )
        test_dataset = SentimentDataset(generators[TEST], cfg.seed_test, cfg)

    elif dataset == LM:
        max_seq_len = cfg.model.max_seq_len
        train_dataset = LanguageModelDataset(
            TRAIN, max_seq_len, cfg.data.dir, generators[TRAIN], seed=cfg.seed
        )
        eval_dataset = LanguageModelDataset(
            EVAL, max_seq_len, cfg.data.dir, generators[EVAL], seed=cfg.seed_eval
        )
        test_dataset = LanguageModelDataset(
            TEST, max_seq_len, cfg.data.dir, generators[TEST], seed=cfg.seed_test
        )
    elif dataset == LR:
        train_dataset = LRDataset(TRAIN, generators[TRAIN], cfg.seed, cfg)
        eval_dataset = LRDataset(EVAL, generators[EVAL], cfg.seed_eval, cfg)
        test_dataset = LRDataset(TEST, generators[TEST], cfg.seed_test, cfg)

    elif dataset == AR:
        train_dataset = ARDataset(
            vocab_size=cfg.data.vocab_size,
            num_xy_pairs=cfg.data.num_xy_pairs_train,
            generators=generators[TRAIN],
            seed=cfg.seed,
            cfg=cfg,
        )
        eval_dataset = ARDataset(
            vocab_size=cfg.data.vocab_size,
            num_xy_pairs=cfg.data.num_xy_pairs_val,
            generators=generators[EVAL],
            seed=cfg.seed_eval,
            cfg=cfg,
        )
        test_dataset = ARDataset(
            vocab_size=cfg.data.vocab_size,
            num_xy_pairs=cfg.data.num_xy_pairs_val,
            generators=generators[TEST],
            seed=cfg.seed_test,
            cfg=cfg,
        )
    elif dataset in [MCC, GMM]:
        d = {MCC: MCCDataset, GMM: GMMDataset}
        mcc = d[dataset]
        train_dataset = mcc(
            num_xy_pairs=cfg.data.num_xy_pairs_train,
            generators=generators[TRAIN],
            seed=cfg.seed,
            cfg=cfg,
        )
        eval_dataset = mcc(
            num_xy_pairs=cfg.data.num_xy_pairs_val,
            generators=generators[EVAL],
            seed=cfg.seed_eval,
            cfg=cfg,
        )
        test_dataset = mcc(
            num_xy_pairs=cfg.data.num_xy_pairs_val,
            generators=generators[TEST],
            seed=cfg.seed_test,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    return train_dataset, eval_dataset, test_dataset


def build_eval_dataset(dataset: str) -> Dataset:
    """Returns the pytorch dataset used to evaluate the model."""
    raise NotImplementedError


def train_loop(
    train_iter,
    dataloader_train,
    model,
    tables,
    # train_dataset,
    dataloader_eval,
    dataloader_test,
    collate_fn,
    optimizer,
    scaler,
    ctx,
    cfg,
    best_eval_loss,
    best_eval_acc,
    patience_left,
):
    loss_fn = build_loss_fn(cfg, in_eval=False)
    exit_training = False
    for _, batch in enumerate(tqdm(dataloader_train, total=cfg.train.iters)):
        # termination condition
        if train_iter == cfg.train.iters:
            exit_training = True
            break
        # reset training
        if (cfg.train.reset_every is not None) and (
            train_iter % cfg.train.reset_every == 0
        ):
            train_iter += 1
            break
        # log training examples to wandb
        rem = train_iter % cfg.eval.every
        if rem in range(cfg.examples_to_log):
            add_example_to_table(
                batch=batch,
                tables=tables,
                split=TRAIN,
                in_eval=False,
                train_iter=train_iter,
                cfg=cfg,
            )
        # evaluate
        if cfg.eval.do and (train_iter % cfg.eval.every == 0):
            model.eval()

            # which splits to evaluate on
            split2data = {
                # TRAIN: train_dataset,
                EVAL: dataloader_eval,
                TEST: dataloader_test,
            }

            e_log = {}
            for split, dataloader in split2data.items():
                dataloader.dataset.reset()
                eval_log = evaluate(
                    model,
                    dataloader=dataloader,
                    ctx=ctx,
                    train_iter=train_iter,
                    collate_fn=collate_fn,
                    split=split,
                    tables=tables,
                    cfg=cfg,
                )
                e_log[f"eval-{split}"] = eval_log

                # if split == "eval" and eval_log['loss'] < best_eval_loss:
                #     torch.save(model.state_dict(), cfg.model['checkpoint_path'])

                # if (
                #     cfg.train.do_save
                #     and (split == "eval")
                #     and (
                #         eval_log["loss"] < best_eval_loss
                #     )  # TODO how to include best_eval_loss
                # ):
                #     # TODO save checkpoints to a dir named after the wandb run id
                #     # so we can easily find it later
                #     best_eval_loss = eval_log["loss"]
                #     torch.save(model.state_dict(), cfg.train.save_path)

            # keep track of best eval accuracy, log its associated test acc
            curr_eval_acc = e_log["eval-eval"].get("acc")
            if curr_eval_acc is not None and curr_eval_acc > best_eval_acc:
                best_eval_acc = curr_eval_acc
                e_log["best_test_acc"] = e_log["eval-test"]["acc"]
                e_log["best_test_acc_train_iter"] = train_iter

            curr_eval_loss = e_log["eval-eval"].get("loss")

            # early stopping
            if cfg.train.early_stop_metric == "loss":
                decr_patience = (
                    curr_eval_loss + cfg.train.early_stop_tol > best_eval_loss
                )
            elif cfg.train.early_stop_metric == "acc":
                if curr_eval_acc is None:
                    raise ValueError(
                        "early_stop_metric is `acc` but curr_eval_acc is `None`"
                    )
                decr_patience = curr_eval_acc - cfg.train.early_stop_tol < best_eval_acc
            else:
                raise ValueError(
                    f"Unknown early_stop_metric: {cfg.train.early_stop_metric}"
                )

            if decr_patience and train_iter >= cfg.train.early_stop_start_iter:
                patience_left -= 1
                logger.info(
                    f"current eval loss {curr_eval_loss:.4f} + tol "
                    f"{cfg.train.early_stop_tol} > best eval loss "
                    f"{best_eval_loss:.4f}, decrementing patience "
                    f"to {patience_left}."
                )
            else:
                patience_left = cfg.train.early_stop_patience
            e_log["patience_left"] = patience_left

            # if eval loss improved, track its associated test loss
            if (curr_eval_loss is not None) and (curr_eval_loss < best_eval_loss):
                best_eval_loss = curr_eval_loss
                if cfg.data._name_ != LM_HF:
                    e_log["best_test_loss"] = e_log["eval-test"]["loss"]
                e_log["best_test_loss_train_iter"] = train_iter

            # need to copy table for wandb to update table as we train
            # e_log[TRAIN + "_table"] = copy.copy(tables[TRAIN])
            # e_log[EVAL_TRAIN + "_table"] = copy.copy(tables[EVAL_TRAIN])
            # e_log[EVAL_EVAL + "_table"] = copy.copy(tables[EVAL_EVAL])
            # e_log[EVAL_TEST + "_table"] = copy.copy(tables[EVAL_TEST])
            wandb.log(
                data=e_log,
                step=train_iter,
            )
            for k, v in e_log.items():
                if isinstance(v, dict):
                    v.pop("per_token_loss", None)
                    v.pop("per_token_acc", None)
                logger.info(f"{k}: {v}")

            if cfg.train.do_early_stop and patience_left == 0:
                logger.info(f"Early stopping at iter {iter}.")
                exit_training = True
                break

            if (cfg.train.early_stop_acc is not None) and (curr_eval_acc is not None):
                if curr_eval_acc > cfg.train.early_stop_acc:
                    logger.info(
                        f"Early stopping at iter {iter} because eval acc is {curr_eval_acc}."
                    )
                    exit_training = True
                    break

            model.train()
            if cfg.save_checkpoints and (train_iter % cfg.train.save_every == 0) and (train_iter != 0):
                save_checkpoint(cfg, model, train_iter, curr_eval_loss)
        ### end evaluation

        # skip training if we are only evaluating
        if not cfg.train.do:
            exit_training = True
            logger.info("Skipping training because cfg.train.do is False.")
            break

        # decay learning rate
        # TODO switch to lr_sceduler like in safari
        lr = (
            get_lr(it=train_iter, cfg=cfg)
            if cfg.scheduler.decay_lr
            else cfg.optimizer.lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # move training data to device
        batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
        targets = batch["targets"]

        # forward pass
        with ctx:
            embeddings = model.embedder(batch)  # [batch_size, seq_len, emb_dim]
            hidden_states = model(embeddings)  # [batch_size, seq_len, emb_dim]
            logits = model.head(hidden_states)  # [batch_size, seq_len, num_classes]
            loss = loss_fn(logits, targets, batch)  # [seq_len] or [seq_len / 2]

        loss_mean = loss.mean()  # [1]
        # backward pass
        scaler.scale(loss_mean).backward()
        grad_clip = cfg.train.grad_clip
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # log stats for current training batch
        if train_iter % cfg.train.log_every == 0:
            # TODO refactor
            if cfg.data._name_ != LR:
                train_log = {"loss": loss_mean.item(), "lr": lr}
                wandb.log(data={"train": train_log}, step=train_iter)
            else:
                # TODO might be able to remove this if
                # we are happy with simple linear regression
                baseline_loss = (
                    sum(
                        max(
                            dataloader_train.dataset.curriculum.n_dims_truncated - ii, 0
                        )
                        for ii in range(dataloader_train.dataset.curriculum.n_points)
                    )
                    / dataloader_train.dataset.curriculum.n_points
                )
                train_log = {
                    "loss": loss_mean.item(),
                    "excess_loss": loss_mean.item() / baseline_loss,
                    "n_points": dataloader_train.dataset.curriculum.n_points,
                    "n_dims": dataloader_train.dataset.curriculum.n_dims_truncated,
                    # sanity checks
                    "_n_points": batch["inputs"].shape[1] / 2,
                    "_n_dims": torch.sum(batch["inputs"][:, 0, :] != 0).item()
                    / batch["inputs"].shape[0],
                    "lr": lr,
                }

                wandb.log(
                    train_log,
                    step=train_iter,
                )

            logger.info(f"train iter {train_iter}")
            for k, v in train_log.items():
                logger.info(f"{k}: {v}")
            logger.info("\n")

        if cfg.data._name_ == LR:
            dataloader_train.dataset.curriculum.update()

        train_iter += 1
    return train_iter, exit_training, best_eval_loss, best_eval_acc


def save_checkpoint(cfg, model, train_iter, eval_loss):
    checkpoint = {
        "cfg": cfg,
        "model_state_dict": model.state_dict(),
        "train_iter": train_iter,
    }
    task = cfg.data._name_
    arch = cfg.model._name_

    # save config
    fp = Path(
        cfg.run_path,
        f"{task}_{arch}_loss{eval_loss:.2f}_iter{train_iter}.pt",
    )

    torch.save(checkpoint, fp)
    logger.info(f"Saved checkpoint to {fp}")


def nl_icl(cfg):
    print(cfg)
    if cfg.nl_icl.checkpoint_path is not None and cfg.nl_icl.hf_path is not None:
        raise ValueError("Must specify either `checkpoint_path` or `hf_path`, not both")

    if cfg.nl_icl.checkpoint_path is not None:
        # Load model from checkpoint
        checkpoint = torch.load(cfg.nl_icl.checkpoint_path)
        cfg.model = checkpoint["cfg"].model
        new_do_flip_class = cfg.data.sent.do_flip_class
        cfg.data = checkpoint["cfg"].data
        cfg.data.sent.do_flip_class = new_do_flip_class
        embedder = build_embedder(cfg)
        head = build_head(cfg)
        model = build_model(cfg, embedder, head)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif cfg.nl_icl.hf_path is not None:
        model = transformers.AutoModelForCausalLM.from_pretrained(cfg.nl_icl.hf_path)
        cfg.data.tokenizer_dir = cfg.nl_icl.hf_path
    else:
        raise ValueError("Must specify either `checkpoint_path` or `hf_path`")
    model.to("cuda")
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.data.tokenizer_dir)

    # Evaluate
    avg_accs = []
    ex_class = []
    seeds = range(cfg.nl_icl.n_seeds)
    examples_per_class = range(
        cfg.nl_icl.min_examples_per_class, cfg.nl_icl.max_examples_per_class + 1
    )

    generators = build_rand_generators(cfg)

    if not cfg.nl_icl.do_full_vocab:
        # figure out what tokens correspond to ' happy' and ' sad'
        happy = tokenizer.encode(" happy", add_special_tokens=False)
        sad = tokenizer.encode(" sad", add_special_tokens=False)

    for n in tqdm(examples_per_class):
        accs = []
        for seed in seeds:
            set_seed(seed)
            cfg.data.sent.examples_per_class = n
            dataset = SentimentDataset(generators[TEST], cfg.seed_test, cfg)
            collate_fn = None
            dataloader = build_dataloader(dataset, collate_fn, TEST, cfg)

            correct = []
            preds = []
            _inputs = []

            for batch in dataloader:
                batch = {
                    k: v.to(cfg.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in batch.items()
                }
                if cfg.nl_icl.checkpoint_path is not None:
                    embeddings = model.embedder(batch)  # [batch_size, seq_len, emb_dim]
                    hidden_states = model(embeddings)  # [batch_size, seq_len, emb_dim]
                    logits = model.head(
                        hidden_states
                    )  # [batch_size, seq_len, num_classes]
                elif cfg.nl_icl.hf_path is not None:
                    logits = model(batch["inputs"]).logits

                if cfg.nl_icl.do_full_vocab:
                    pred = torch.argmax(logits[:, -1, :], dim=-1)  # [batch_size]
                    pred = tokenizer.decode(pred, skip_special_tokens=True)
                    if batch["dec_targets"][0] in pred:
                        correct.append(1)
                    else:
                        correct.append(0)
                else:
                    # zero out logits that don't correspond to ' happy' or ' sad'
                    # start by creating a tensor with all zeros and the same shape as logits
                    mask = torch.zeros_like(logits)
                    # then set the values at the indices corresponding to ' happy' and ' sad' to 1
                    mask[:, -1, happy] = 1
                    mask[:, -1, sad] = 1
                    # then multiply the mask by the logits
                    logits = logits * mask
                    pred = torch.argmax(logits[:, -1, :], dim=-1)  # [batch_size]
                    pred = pred.item()
                    if batch["dec_targets"][0] == "happy":
                        correct.append(1 if pred == happy[0] else 0)
                    elif batch["dec_targets"][0] == "sad":
                        correct.append(1 if pred == sad[0] else 0)
                    pred = tokenizer.decode(pred, skip_special_tokens=True)

                # pred = tokenizer.decode(pred, skip_special_tokens=True)
                preds.append(pred)

                _inputs.append(batch["dec_inputs"])
            accs.append(sum(correct) / len(correct))
        avg_acc = sum(accs) / len(accs)

        for i, (prompt, pred) in enumerate(zip(_inputs, preds)):
            if i == 3:
                break
            print(f"{i} {prompt=}")
            print(f"{i} {pred=}")

        # print(preds)
        # print(f"{accs=}")
        print(f"Examples per class: {n}, Accuracy: {avg_acc}")
        print()
        avg_accs.append(avg_acc)
        ex_class.append(n)

    fp = f"nl_icl/do_flip_class={cfg.data.sent.do_flip_class}-do_full_vocab={cfg.nl_icl.do_full_vocab}.jsonl"
    with jsonlines.open(fp, mode="a") as writer:
        writer.write(
            {
                "model": wandb.config.model
                if cfg.nl_icl.hf_path is None
                else {"_name_": cfg.nl_icl.hf_path},
                # "config": cfg.nl_icl,
                "avg_accs": avg_accs,
                "ex_class": ex_class,
            }
        )


def train(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    test_dataset: Dataset,
    tables: dict,
    cfg: DictConfig,
):
    # automatic mixed precision
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.train.dtype]
    ctx = (
        nullcontext()
        if cfg.device == "cpu"
        else torch.amp.autocast(
            device_type=cfg.device, dtype=ptdtype, enabled=cfg.train.do_amp
        )
    )
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.dtype == "float16"))

    model.train()
    model.to(cfg.device)

    # monitor gradients
    wandb.watch(model, log_freq=cfg.train.log_every)

    collate_fn = None
    dataloader_train = build_dataloader(train_dataset, collate_fn, TRAIN, cfg)
    dataloader_eval = build_dataloader(eval_dataset, collate_fn, EVAL, cfg)
    dataloader_test = build_dataloader(test_dataset, collate_fn, TEST, cfg)
    optimizer, lr_scheduler = build_optimizer(cfg=cfg, model=model)

    # start train loop
    best_eval_loss = float("inf")
    best_eval_acc = float("-inf")
    train_iter = 0
    patience_left = cfg.train.early_stop_patience
    while True:
        train_iter, exit_training, best_eval_loss, best_eval_acc = train_loop(
            train_iter=train_iter,
            dataloader_train=dataloader_train,
            model=model,
            tables=tables,
            # train_dataset=train_dataset,
            dataloader_eval=dataloader_eval,
            dataloader_test=dataloader_test,
            collate_fn=collate_fn,
            optimizer=optimizer,
            scaler=scaler,
            ctx=ctx,
            cfg=cfg,
            best_eval_loss=best_eval_loss,
            best_eval_acc=best_eval_acc,
            patience_left=patience_left,
        )
        if exit_training:
            break
        else:
            logger.info(f"Resetting training at train iter {train_iter}.")
    return


@torch.no_grad()
def evaluate(
    model: nn.Module,
    # metrics: dict[str, callable],
    dataloader: Dataset,
    ctx,
    train_iter: int,
    collate_fn,
    split: str,
    tables: dict,
    cfg: DictConfig,
) -> dict:
    model.eval()
    data_name = cfg.data._name_
    eval_iters = cfg.eval.iters
    loss_fn = build_loss_fn(cfg, in_eval=True)

    ### 1. init storage for metrics across iterations
    per_token_loss = []  # unreduced loss
    if data_name == LR:
        oracle_losses = []
    elif data_name in [OG, AR, MCC, GMM]:
        n_correct = []
        n_total = torch.zeros(eval_iters)
    elif data_name == LM_HF:
        tokenizer = dataloader.dataset.tokenizer
        n_correct = []
        n_total = torch.zeros(eval_iters)
        prompts = []
        preds = []
    # else:
    #     raise ValueError(f"Invalid data name: `{data_name}`")

    ### 2. evaluate
    for iter, batch in enumerate(tqdm(dataloader, leave=False, total=eval_iters)):
        if iter == eval_iters:
            break
        # log eval examples to wandb
        if iter in range(cfg.examples_to_log):
            add_example_to_table(
                batch=batch,
                tables=tables,
                split=split,
                in_eval=True,
                train_iter=train_iter,
                cfg=cfg,
                eval_iter=iter,
            )
        # move to device
        batch = {
            k: v.to(cfg.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        targets = batch["targets"]
        # forward pass
        with ctx:
            embeddings = model.embedder(batch)  # [batch_size, seq_len, emb_dim]
            hidden_states = model(embeddings)  # [batch_size, seq_len, emb_dim]
            logits = model.head(hidden_states)  # [batch_size, seq_len, num_classes]
            # loss shape depen on loss_fn. usually [seq_len]
            loss = loss_fn(logits, targets, batch, cfg.train.merge_embeds)

        # compute metrics over batch
        per_token_loss.append(loss)

        if data_name == OG:
            # accuracy
            # take last logits
            last_logits = logits[:, -1, :]  # [batch_size, num_classes]
            # only consider first two classes
            if split in [EVAL, TEST]:
                last_logits = last_logits[:, :2]  # [batch_size, 2]
            pred = torch.argmax(last_logits, dim=-1)  # [batch_size]
            last_targets = targets[:, -1]  # [batch_size]
            correct = pred == last_targets  # [batch_size]
            correct = rearrange(correct, "b -> b 1")
            n_correct.append(correct)
            n_total[iter] = targets.shape[0]

        elif data_name in [AR, MCC, GMM]:
            # logits [b, 2*s-1, c]
            # targets [b, s]
            if cfg.train.merge_embeds:
                last_logits = logits[:, -1, :]
                pred = torch.argmax(last_logits, dim=-1)  # [b, s]
                last_targets = targets[:, -1]  # [b, s]
                correct = pred == last_targets
                correct = rearrange(correct, "b -> b 1")
            else:
                logits = logits[:, ::2, :]  # [b, s, c]
                pred = torch.argmax(logits, dim=-1)  # [b, s]
                correct = pred == targets  # [b, s]
            n_correct.append(correct)
            n_total[iter] = targets.shape[0]

        elif data_name == LR:
            # compute oracle predictions by averaging past y values
            oracle_pred = running_average(targets)  # [b, s/2]
            # prepend 0, drop last
            oracle_pred = torch.cat(
                [torch.zeros_like(oracle_pred[:, :1]), oracle_pred[:, :-1]], dim=1
            )
            oracle_loss = F.mse_loss(oracle_pred, targets, reduction="none")  # [b, s/2]
            oracle_losses.append(oracle_loss)

        elif data_name == LM_HF and split == TEST:
            pred = torch.argmax(logits[:, -1, :], dim=-1)  # [batch_size]
            pred = tokenizer.decode(pred, skip_special_tokens=True)
            if batch["dec_targets"][0] in pred:
                n_correct.append(1)
            else:
                n_correct.append(0)
            prompts.append(batch["dec_inputs"][0])
            preds.append(pred)

    ### 3. summarize metrics after eval loop
    log = {}
    # convert from list of tensors to tensor
    if not (data_name == LM_HF and split == TEST):
        per_token_loss = rearrange(per_token_loss, "t b s -> (t b) s")
        log["loss"] = per_token_loss.mean().item()
        per_token_loss = reduce(per_token_loss, "b s -> s", "mean")
        log["per_token_loss"] = per_token_loss.tolist()
        log["last_loss"] = per_token_loss[-1].item()

    if data_name in [AR, MCC, GMM, OG]:
        n_correct = rearrange(n_correct, "t b s -> (t b) s")
        n_correct = reduce(n_correct, "b s -> s", "sum")
        n_total = torch.sum(n_total)
        per_token_acc = n_correct / n_total  # [s]
        log["per_token_acc"] = per_token_acc.tolist()
        log["avg_acc"] = per_token_acc.mean().item()
        log["acc"] = per_token_acc[-1].item()  # [1]

    elif (data_name == LM) or (data_name == LM_HF and split == EVAL):
        # compute ICL score
        early_token_loss = per_token_loss[40:60]
        long_token_loss = per_token_loss[490:510]
        icl_score = long_token_loss - early_token_loss
        log["long_icl_score"] = icl_score.mean().item()

        short_token_loss = per_token_loss[190:210]
        icl_score = short_token_loss - early_token_loss
        log["short_icl_score"] = icl_score.mean().item()

        save_path = Path(cfg.run_path, f"per_token_loss_{split}.jsonl")
        with open(save_path, "a") as f:
            row = {"train_iter": train_iter, "per_token_loss": per_token_loss.tolist()}
            f.write(json.dumps(row) + "\n")

    elif data_name == LR:
        oracle_losses = rearrange(oracle_losses, "t b s -> (t b) s")
        oracle_losses = reduce(oracle_losses, "b s -> s", "mean")
        log["oracle_loss"] = oracle_losses[-1].item()

    if data_name == LM_HF and split == TEST:
        log["acc"] = sum(n_correct) / len(n_correct)
        # log the first 3 prompts and their predictions
        logger.info(f"Prompts: {prompts[:3]}")
        logger.info(f"Predictions: {preds[:3]}")

    model.train()
    return log


def prepare_images_for_wandb(logits, batch, step):
    images = batch["example"]
    targets = batch["targets"]
    target = batch["target"]
    images_per_row = images.shape[1]
    # collapse batch and seq_len dimensions
    images = images.view(-1, *images.shape[2:])
    # move channels to second dimension
    images = images.permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(images, nrow=images_per_row)
    preds = logits.argmax(dim=-1)
    caption = (
        f"labels: {batch['label'].tolist()}\n"
        f"preds: {preds.tolist()}\n"
        f"targets: {targets.tolist()}\n"
        f"target: {target.tolist()}\n"
        f"step: {step}"
    )

    return {"image": wandb.Image(grid, caption=caption)}


def build_optimizer(model, cfg):
    model_name = cfg.model._name_
    optimizer_name = cfg.optimizer._name_
    optimizer = lr_scheduler = None
    # if model_name == TRAN:
    #     optimizer = model.configure_optimizers(
    #         cfg.optimizer.weight_decay,
    #         cfg.optimizer.lr,
    #         cfg.optimizer.betas,
    #         cfg.device,
    #     )
    # # elif model_name == S4:
    # #     optimizer = model.setup_optimizer(
    # #         lr=wandb.config.train["learning_rate"],
    # #         weight_decay=wandb.config.train["weight_decay"],
    # #     )
    # elif model_name == SAFARI:
    #     optimizer, lr_scheduler = model.configure_optimizers(cfg=cfg)
    #     optimizer = optimizer[0]
    #     lr_scheduler = lr_scheduler[0]
    # else:
    # print(f"Model `{model}` not recognized. Defaulting to AdamW.")
    # TODO add more optimizers
    # TODO list out which parameters should/should not have weight decay.
    # Will depend on architecture
    if optimizer_name == "adamw":
        # TODO adam breaks rwkv + custom cuda kernel. why?
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            # fused=True,  # TODO breaks s4 in safari
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            # momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif optimizer_name == "nesterov":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=True,
        )
    else:
        raise NotImplementedError
    return optimizer, lr_scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:
    # logging
    if cfg.log_level == "info":
        transformers.logging.set_verbosity_info()
    elif cfg.log_level == "warning":
        transformers.logging.set_verbosity_warning()
    else:
        raise ValueError(f"Invalid log level: {cfg.log_level}")

    # enable adding new keys to cfg
    OmegaConf.set_struct(cfg, False)

    # set training iters based on samples and batch size
    if cfg.train.samples is not None:
        train_iters = cfg.train.samples // cfg.train.batch_size
        cfg.train.iters = train_iters - (cfg.train.samples % cfg.train.batch_size) + 1

    # update eval.every based on eval.every_samples and batch size
    if cfg.eval.every_samples is not None:
        eval_every = cfg.eval.every_samples // cfg.train.batch_size
        cfg.eval.every = eval_every - (cfg.eval.every_samples % cfg.eval.batch_size)

    # log config to wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project)
    wandb.config.update(wandb_cfg, allow_val_change=True)

    # track best metrics
    wandb.define_metric("eval-eval.acc", summary="max")
    wandb.define_metric("eval-train.acc", summary="max")
    wandb.define_metric("eval-test.acc", summary="max")
    wandb.define_metric("eval-eval.loss", summary="min")

    # track test metrics associated with best eval metrics
    wandb.define_metric("best_test_acc", summary="max")
    wandb.define_metric("best_test_loss", summary="min")
    # track their corresponding training iteration
    wandb.define_metric("best_test_acc_train_iter", summary="last")
    wandb.define_metric("best_test_loss_train_iter", summary="last")

    wandb.define_metric("loss", summary="min")
    # pretty print config
    utils.train.print_config(cfg, resolve=True)
    # reproducibility
    set_seed(cfg.seed)

    if cfg.nl_icl.do:
        nl_icl(cfg)
        return

    generators = build_rand_generators(cfg)
    train_dataset, eval_dataset, test_dataset = build_datasets(generators, cfg)
    embedder = build_embedder(cfg)
    head = build_head(cfg)
    model = build_model(cfg, embedder, head)

    # count parameters
    count_parameters(model)
    if cfg.do_count_param_only:
        return

    # create local directory specific to this run
    run_path = Path(cfg.train.save_path, wandb.run.id)
    run_path.mkdir(parents=True, exist_ok=True)
    cfg.run_path = str(run_path)

    # save config
    cfg_fp = Path(cfg.run_path, "cfg.yaml")
    OmegaConf.save(cfg, cfg_fp)

    # tables for logging training and eval examples to wandb
    tables = {}
    train_columns = eval_columns = []
    if cfg.data._name_ == LM:
        train_columns = ["train_iter", "batch_idx", "input"]
        eval_columns = ["train_iter", "eval_iter", "batch_idx", "input"]
    else:
        train_columns = ["train_iter", "batch_idx", "input", "target"]
        eval_columns = ["train_iter", "eval_iter", "batch_idx", "input", "target"]
    tables[TRAIN] = wandb.Table(columns=train_columns)
    tables[EVAL_TRAIN] = wandb.Table(columns=eval_columns)
    tables[EVAL_EVAL] = wandb.Table(columns=eval_columns)
    tables[EVAL_TEST] = wandb.Table(columns=eval_columns)

    # Train
    train(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        tables=tables,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
