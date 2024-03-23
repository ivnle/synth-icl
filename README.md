# Is attention required for ICL? Exploring the Relationship Between Model Architecture and In-Context Learning Ability
- https://arxiv.org/abs/2310.08049
- published as a conference paper at ICLR 2024

# install
```
conda create -n icl python=3.10
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
pip install torch==2.0.1
pip install packaging==23.2
pip install -r requirements.txt
```

# sweeps to replicate experiments
- `sweeps/multiclass_classification.yaml` section 4
- `sweeps/linear_regression.yaml` section 4
- `sweeps/assoc_recall.yaml` section 4
- `sweeps/omniglot.yaml` image classifcation, section 5
- `sweeps/lang_modeling.yaml` section 6 
- `sweeps/linear_regression_noisy.yaml` noisy linear regression, appendix B.1
- `sweeps/ar_pi-x.yaml` `sweeps/lr_pi-x.yaml` `sweeps/gmm_pi-x.yaml` `sweeps/og_pi-x.yaml` permutation invariance, appendix E

# generate figures
- stats/extrap.ipynb
- stats/og.ipynb
- stats/extrap_lr_noisy.ipynb
- nl_icl/lang_model.ipynb
- nl_icl/nl_icl.ipynb

# ridge and logistic regression baselines
- lr_leastsquares.py
- mcc_oracle.py
