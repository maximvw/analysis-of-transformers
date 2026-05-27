"""Run ablation experiments: SFT baseline vs auxiliary loss configurations.

Usage:
    python -m llm_auxiliary.scripts.run_experiment --mode sft
    python -m llm_auxiliary.scripts.run_experiment --mode bracket
    python -m llm_auxiliary.scripts.run_experiment --mode indent
    python -m llm_auxiliary.scripts.run_experiment --mode mode
    python -m llm_auxiliary.scripts.run_experiment --mode all
"""

from __future__ import annotations

import argparse

from llm_auxiliary.src.train import TrainConfig, train


ABLATION_CONFIGS = {
    "sft": {
        "lambda_bracket": 0.0,
        "lambda_indent": 0.0,
        "lambda_mode": 0.0,
        "output_dir": "checkpoints/llm_sft",
    },
    "bracket": {
        "lambda_bracket": 1.0,
        "lambda_indent": 0.0,
        "lambda_mode": 0.0,
        "output_dir": "checkpoints/llm_bracket",
    },
    "indent": {
        "lambda_bracket": 0.0,
        "lambda_indent": 1.0,
        "lambda_mode": 0.0,
        "output_dir": "checkpoints/llm_indent",
    },
    "mode": {
        "lambda_bracket": 0.0,
        "lambda_indent": 0.0,
        "lambda_mode": 1.0,
        "output_dir": "checkpoints/llm_mode",
    },
    "all": {
        "lambda_bracket": 1.0,
        "lambda_indent": 1.0,
        "lambda_mode": 1.0,
        "output_dir": "checkpoints/llm_all",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Experiment mode: sft, bracket, indent, mode, all",
    )
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=50_000)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ablation = ABLATION_CONFIGS[args.mode]

    config = TrainConfig(
        model_name=args.model_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        seq_length=args.seq_length,
        load_in_8bit=args.load_in_8bit,
        seed=args.seed,
        **ablation,
    )

    print(f"=== Experiment: {args.mode} ===")
    print(f"  lambda_bracket={config.lambda_bracket}")
    print(f"  lambda_indent={config.lambda_indent}")
    print(f"  lambda_mode={config.lambda_mode}")
    print(f"  output_dir={config.output_dir}")
    print()

    train(config)


if __name__ == "__main__":
    main()
