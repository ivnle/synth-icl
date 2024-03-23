optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "arch.safari.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "arch.safari.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "arch.safari.models.sequence.SequenceModel",
    "lm": "arch.safari.models.sequence.long_conv_lm.ConvLMHeadModel",
    "lm_simple": "arch.safari.models.sequence.simple_lm.SimpleLMHeadModel",
    "vit_b_16": "arch.safari.models.baselines.vit_all.vit_base_patch16_224",
}

layer = {
    "id": "arch.safari.models.sequence.base.SequenceIdentity",
    "ff": "arch.safari.models.sequence.ff.FF",
    "mha": "arch.safari.models.sequence.mha.MultiheadAttention",
    "s4d": "arch.safari.models.sequence.ssm.s4d.S4D",
    "s4_simple": "arch.safari.models.sequence.ssm.s4_simple.SimpleS4Wrapper",
    "long-conv": "arch.safari.models.sequence.long_conv.LongConv",
    "h3": "arch.safari.models.sequence.h3.H3",
    "h3-conv": "arch.safari.models.sequence.h3_conv.H3Conv",
    "hyena": "arch.safari.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "arch.safari.models.sequence.hyena.HyenaFilter",
    "vit": "arch.safari.models.sequence.mha.VitAttention",
}

callbacks = {
    "timer": "arch.safari.callbacks.timer.Timer",
    "params": "arch.safari.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "arch.safari.callbacks.progressive_resizing.ProgressiveResizing",
}
