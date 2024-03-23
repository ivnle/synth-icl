# #!/bin/bash

FLIP=false
FULL_VOCAB=false

# FLIP=false
# FULL_VOCAB=true

# FLIP=true
# FULL_VOCAB=false

# FLIP=true
# FULL_VOCAB=true


list1=(
    # "rnn"
    # "lstm"
    "gru"
    "lightconv"
    "dynamicconv"
    "s4"
    "h3"
    "hyena"
    "mamba"
    "retnet"
    "rwkv"
    "gpt2"
    "llama2"
)

list2=(
    # "/trunk/checkpoints/ivanlee/icl-arch/a52qijrn/language-modeling-HF_rnn_loss2.25_iter200000.pt"
    # "/trunk/checkpoints/ivanlee/icl-arch/gw1lk6dq/language-modeling-HF_lstm_loss1.44_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_gru_ar52htrh/language-modeling-HF_gru_loss1.50_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_lightconv_ccufglux/language-modeling-HF_lightconv_loss1.48_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_dynamicconv_vfvjy4i7/language-modeling-HF_dynamicconv_loss1.42_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_s4_e4dkk56k/language-modeling-HF_safari_loss1.39_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_h3_klyeu3yc/language-modeling-HF_safari_loss1.35_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_hyena_v2l4ykuh/language-modeling-HF_safari_loss1.33_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_mamba_0sncxu66/language-modeling-HF_mamba_loss1.30_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_retnet_9n65h4xw/language-modeling-HF_retnet_loss1.38_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_rwkv_zx7wwkt5/language-modeling-HF_rwkv_loss1.32_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_gpt2_o2khttk2/language-modeling-HF_gpt2_loss1.32_iter200000.pt"
    "/graft3/checkpoints/ivanlee/icl-arch/lm_llama2_ii4pewcb/language-modeling-HF_llama2_loss1.30_iter200000.pt"
)

# Get the length of the lists
length1=${#list1[@]}
length2=${#list2[@]}

# Use the smaller length for the loop
length=$((length1 < length2 ? length1 : length2))

# Loop over the indices of the shortest list
for ((i=1; i<$length+1; i++)); do
    # Access the elements from each list using the index
    item1=${list1[$i]}
    item2=${list2[$i]}

    python main.py log_level=warning data=lang-model model=${item1} nl_icl.do=True nl_icl.checkpoint_path=${item2} nl_icl.min_examples_per_class=0 nl_icl.max_examples_per_class=9 nl_icl.n_seeds=10 data.sent.do_flip_class=$FLIP nl_icl.do_full_vocab=$FULL_VOCAB
done
