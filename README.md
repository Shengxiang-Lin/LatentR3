Reinforced Latent Reasoning for LLM-based Recommendation

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

Latent  Dataset:CDs_and_Vinyl    
|  Model |Sample|NG@1|NG@3|NG@5|NG@10|HR@1|HR@3|HR@5|HR@10|
|--------|------|----|----|----|----|----|----|----|----|
|Qwen2.5-1.5B-Instruct|    0|0.0050|0.0085|0.0100|0.0122|0.0050|0.0112|0.0148|0.0216|
|Qwen2.5-1.5B-Instruct| 1000|0.0083|0.0093|0.0103|0.0128|0.0083|0.0103|0.0127|0.0205|
|Qwen2.5-1.5B-Instruct| 1000|0.00081|0.0015|0.0020|0.0029|0.00081|0.0021|0.0032|0.0060|
|Qwen2.5-1.5B-Instruct| 1000|0.0099|0.0115|0.0126|0.0139|0.0099|0.0127|0.0153|0.0193|
|Qwen2.5-1.5B-Instruct| 1000|0.0180|0.0235|0.0260|0.0283|0.0180|0.0276|0.0335|0.0408|
|Qwen2.5-1.5B-Instruct| 5000|0.0122|0.0153|0.0164|0.0190|0.0122|0.0177|0.0203|0.0284|
|Qwen2.5-1.5B-Instruct| 5000|0.0224|0.0268|0.0282|0.0312|0.0224|0.0302|0.0335|0.0429|
|Qwen2.5-1.5B-Instruct|10000|0.0304|0.0356|0.0377|0.0406|0.0304|0.0391|0.0445|0.0534|
|Qwen2.5-1.5B-Instruct|10000|0.0274|0.0328|0.0347|0.0374|0.0274|0.0369|0.0414|0.0499|
|Qwen2.5-1.5B-Instruct|49251|0.0482|0.0582|0.0625|0.0673|0.0482|0.0653|0.0757|0.0905|

GRPO based on ALL Latent
|  Model |  lr  | Sample|NG@1|NG@3|NG@5|NG@10|HR@1|HR@3|HR@5|HR@10|
|--------|------|----|----|----|----|----|----|----|----|----|
|Qwen2.5-1.5B-Instruct|5e-4|49251|0.00049|0.00083|0.00083|0.00083|0.00049|0.00114|0.00114|0.00114|
|Qwen2.5-1.5B-Instruct|5e-5|49251|0.0512|0.0617|0.0658|0.0704|0.0512|0.0690|0.0791|0.0934|
|Qwen2.5-1.5B-Instruct|1e-4|49251|0.0486|0.0589|0.0631|0.0679|0.0486|0.0663|0.0763|0.0911|
|Qwen2.5-1.5B-Instruct|1e-4|49251|0.0484|0.0585|0.0631|0.0677|0.0484|0.0656|0.0768|0.0910|

GRPO_Group based on ALL Latent
|  Model |  lr  | Sample|NG@1|NG@3|NG@5|NG@10|HR@1|HR@3|HR@5|HR@10|
|--------|------|----|----|----|----|----|----|----|----|----|
|Qwen2.5-1.5B-Instruct|1e-4|49251|0.0552|0.0666|0.0715|0.0762|0.0552|0.0747|0.0867|0.1012|