# Fine-tune LLMs on MacBook with MLX

Fine-tune Large Language Models on Apple Silicon Macs using MLX framework and LoRA. This repository demonstrates medical diagnosis model fine-tuning using MLX's efficient training capabilities. Check the Medium article for a step by step walkthrough of the code. 

https://medium.com/@dummahajan/train-your-own-llm-on-macbook-a-15-minute-guide-with-mlx-6c6ed9ad036a

## Requirements
- MacBook with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX framework
- ~16GB RAM

## Installation
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U mlx-lm
pip install -r requirements.txt
```

## Project Structure
```
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── valid.jsonl
├── scripts/
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Quick Start

1. Setup Hugging Face credentials:
```bash
huggingface-cli login  # Requires access token from huggingface.co/settings/tokens
```

2. Prepare dataset:
```bash
python scripts/preprocess.py --input dataset/symptoms_diagnosis.csv
```

3. Fine-tune model:
```bash
python -m mlx_lm.lora \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --data data \
    --train \
    --fine-tune-type lora \
    --batch-size 4 \
    --num-layers 16 \
    --iters 1000 \
    --adapter-path adapters
```

4. Fuse model with adapters:
```bash
python -m mlx_lm.fuse \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --adapter-path adapters \
    --save-path model/fine-tuned_model
```

## Features
- PEFT/LoRA implementation
- Quantized model support
- Custom dataset preprocessing
- Memory-efficient training
- Model fusion capabilities

## Reference
- [MLX Community Models](https://huggingface.co/mlx-community)
- [MLX Examples](https://github.com/ml-explore/mlx-examples/tree/main/lora)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
