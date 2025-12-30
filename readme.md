# Editing *Monet → Van Gogh*

```bash
python train.py \
  --model_id "CompVis/stable-diffusion-v1-4" \
  --target_prompt "Claude Monet" \
  --anchor_prompt "Van Gogh" \
  --time_max 300 \
  --time_min 5 \
  --infer_steps 20 \
  --cfg \
  --guidance_scale 5.0 \
  --num_epochs 50 \
  --batch_size 48 \
  --mixed_precision "bf16" \
  --use_lora \
  --lora_rank 4 \
  --seed 123445 \
  --lr 1e-4 \
  --save_freq 10 \
  --max_grad_norm 1.0 \
  --accumulation_steps 1 \
  --ema 0 \
  --save_dir "./" \
  --wandb "GflowUnlearn" \
  --eval_prompts \
    "San Giorgio Maggiore at Dusk, in style of Claude Monet" \
    "San Giorgio Maggiore at Dusk, in style of Claude Monet" \
    "San Giorgio Maggiore at Dusk, in style of Van Gogh" \
  --eval_freq 1
