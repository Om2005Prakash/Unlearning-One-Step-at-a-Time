# Editing *Monet → Van Gogh*

## Quick Use

```bash
!python gflow_unlearn.py \
--model_id "CompVis/stable-diffusion-v1-4" \
--name "monet_to_vangogh" \
--target_prompt "monet" \
--anchor_prompt "vangogh" \
--paired_prompt_dataset "paried_prompts.csv" \
--time_max 999 \
--time_min 0 \
--infer_steps 20 \
--cfg \
--guidance_scale 5.0 \
--num_epochs 1000 \
--batch_size 8 \
--mixed_precision "bf16" \
--use_lora \
--lora_rank 4 \
--seed 123345 \
--lr 1e-4\
--save_freq 20 \
--max_grad_norm 1.0 \
--accumulation_steps 1 \
--ema 0 \
--save_dir "./unlearned_models" \
--wandb "GFlow Unlearn"\
--eval_freq 1 \
--eval_prompts "San Giorgio Maggiore at Dusk, in style of Monet_San Giorgio Maggiore at Dusk, in style of Monet_San Giorgio Maggiore at Dusk, in style of Van Gogh"
```

## Result GIF:
![Example Result](https://raw.githubusercontent.com/Om2005Prakash/Unlearning-One-Step-at-a-Time/refs/heads/main/out.gif)
