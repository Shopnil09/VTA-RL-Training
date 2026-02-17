"""
Student Simulator RL Training Script

Trains a small LLM (Qwen2.5-0.5B-Instruct) using GRPO to generate
student discussion forum posts that stylistically match target clusters.

This uses TRL's GRPOTrainer for local CPU/MPS testing.
When moving to GPU, switch to prime-rl with a TOML config.

Usage:
    python train.py

Requirements:
    pip install trl transformers datasets torch accelerate peft numpy
"""

import json
import sys
from pathlib import Path
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

# Add project root to path so we can import the environment
sys.path.insert(0, str(Path(__file__).parent))
from environments.student_simulator.student_simulator import (
    style_reward_func,
    init_reward,
)

# Configuration

# Model - smallest Qwen for local testing
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Paths
DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "dataset.json"
CLUSTER_DATA_PATH = DATA_DIR / "cluster_data.pkl"
OUTPUT_DIR = Path(__file__).parent / "output"

# Privacy (inf = no noise for testing)
EPSILON = float("inf")


# Load dataset

def load_dataset_from_json(path: Path) -> Dataset:
    """
    Load dataset from JSON and convert to HuggingFace Dataset.

    TRL GRPOTrainer expects:
      - "prompt" column: list of message dicts OR string
      - Any extra columns (like "cluster_id") are passed as kwargs to reward func
    """
    with open(path, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)

    print(f"Loaded {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print(f"Cluster distribution: { {cid: sum(1 for x in dataset['cluster_id'] if x == cid) for cid in set(dataset['cluster_id'])} }")

    return dataset


# Main training function

def main():
    print("=" * 60)
    print("Student Simulator - GRPO Training")
    print("=" * 60)
    
    # load dataset
    print("\n[1/4] Loading dataset...")
    dataset = load_dataset_from_json(DATASET_PATH)

    # initialize reward function
    print("\n[2/4] Initializing reward function...")
    init_reward(
        cluster_data_path=CLUSTER_DATA_PATH,
        epsilon=EPSILON,
    )

    # configure trainer
    print("\n[3/4] Configuring trainer...")

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),

        # === Batch / step settings (tiny for local testing) ===
        max_steps=5,                        # just 5 steps to verify pipeline
        per_device_train_batch_size=2,      # small batch for CPU
        num_generations=2,                  # completions per prompt (GRPO group size)
        gradient_accumulation_steps=1,

        # === Generation settings ===
        max_completion_length=100,          # short completions for fast testing
        temperature=0.7,

        # === Learning rate ===
        learning_rate=1e-5,

        # === Logging ===
        logging_steps=1,                    # log every step
        report_to="none",                   # disable wandb for local testing

        # === Device ===
        # TRL auto-detects CPU/MPS/CUDA
        # On Mac: runs on CPU (slow but works)
        # bf16/fp16 disabled for CPU compatibility
        bf16=False,
        fp16=False,

        # === Keep extra columns (cluster_id) for reward function ===
        remove_unused_columns=False,

        # === Save ===
        save_steps=999999,                  # don't save checkpoints during test
    )

    # --- Initialize trainer ---
    print("\n[4/4] Starting training...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Steps: {training_args.max_steps}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Generations per prompt: {training_args.num_generations}")
    print(f"  Max completion length: {training_args.max_completion_length}")
    print(f"  Device: CPU (Mac local testing)")
    print()

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=style_reward_func,
        train_dataset=dataset,
        args=training_args,
    )

    # running the trainer
    print("Training started...")
    print("(This will be slow on CPU - just verifying the pipeline works)")
    print()

    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()