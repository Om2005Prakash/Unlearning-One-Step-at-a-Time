import argparse

"""
model_id

target_prompt
anchor_prompt
time_max
time_min
infer_steps
cfg
guidance_scale

num_epochs
batch_size
mixed_precision
use_lora
lora_rank
seed
lr
save_freq
max_grad_norm

ema
save_dir

wandb
"""

def get_args():
    parser = argparse.ArgumentParser(description="EraseFlow‐style LoRA training")

    # Core model
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace repo or local path for Stable Diffusion."
    )

    # Objective Args
    parser.add_argument(
        "--target_prompt",
        type=str,
        required=True,
        help="Prompt whose concept we want to erase."
    )
    parser.add_argument(
        "--anchor_prompt",
        type=str,
        required=True,
        help="Safe prompt used for sampling."
    )
    parser.add_argument(
        "--time_min", 
        type=int, 
        default=0, 
        help=""
    )
    parser.add_argument(
        "--time_max", 
        type=int, 
        default=999, 
        help=""
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="Number of DDPM sampling steps."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Text‐to‐image guidance scale."
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Enable explicit classifier‐free guidance in the UNet forward pass."
    )

    # Training Args
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per training step."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Mixed precision for non‐trainable parts."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to attach LoRA to UNet."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank for LoRA adapters."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for LoRA parameters."
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Save a LoRA checkpoint every N epochs."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping."
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps."
    )

    # Inference Args
    parser.add_argument(
        "--ema",
        type=float,
        default=0,
        help="Exponential Moving Avg"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory under which subdir `--name` will be created."
    )
    
    # Logging
    parser.add_argument(
        "--wandb", 
        type=str, 
        default="Unlearn_SD", 
        help=""
    )
    parser.add_argument(
        "--eval_prompts",
        type=str,
        default="",
        help="Eval Prompts"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        help="Eval frequency"
    )

    # Debug
    parser.add_argument(
        "--test_loss_scale",
        action="store_true",
        help="Loss value when Flow is already satisfied."
    )

    return parser.parse_args()