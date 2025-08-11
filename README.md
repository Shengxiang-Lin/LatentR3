

## Directory Structure

```plaintext
.
|____scripts
|____src
| |____latent
| | (SFT code)
| |____grpo_attention_tuning (RL code)
| | |____latent_grpo_processor.py
| | |____latent_grpo_dataset.py
| | |____model.py
| | |____grpo_trainer.py
| | |____train_noise_grpo.py
| | |____noise_eval.py
| |____utils (others)
|____data (example data and our processing code)
| |____process.py
|____environment.yaml
|____README.md
```

## Environment

```bash
conda env create -f environment.yaml
```

## Train and evaluation

```bash
conda activate trl
# SFT
bash scripts/latent_train.sh
# modified GRPO (change the path to your sft model)
bash scripts/attention_grpo.sh
# for evaluation
bash scripts/grpo_eval.sh
```

> **Note:**
> - We strongly recommend that you use two GPUs to run GRPO. We did not test our code on more GPUs due to computational resources. This is related to the calculation of the advantage function.

## Hyperparameter Search

To achieve expected results, tune the learning rate in GRPO:

- `lr`: {1e-5, 1e-4, 5e-4}

You can tune more hyperparameters to get more impressive results.

> **Note:**
> - Results may vary across different devices even with the same hyperparameters, due to differences in computation precision. We conduct our experiments using 2 NVIDIA A100 GPUs.


CDs_and_Vinyl
1024
[0.00830078 0.00934127 0.01032854 0.0128167  0.0128167 ]
[0.00830078 0.01025391 0.01269531 0.02050781 0.02050781]

[0.00081208 0.00152555 0.00200807 0.00288363 0.00288363]
[0.00081208 0.00211142 0.00324834 0.00600942 0.00600942]

5000
[0.01218126 0.01532512 0.01639448 0.0189625  0.0189625 ]
[0.01218126 0.01770343 0.0203021  0.02842293 0.02842293]

10000

all(49251)
[0.04823778 0.05821069 0.06249525 0.06734953 0.06734953]
[0.04823778 0.06529154 0.07568621 0.09046614 0.09046614]