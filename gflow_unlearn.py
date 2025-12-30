%%writefile train.py

"""
Plan:

1> Take arg time range to perform alignment, also take infer_steps
2> Sample a batch of forget concept trajectories
3> Update
"""

import os
import glob
from pathlib import Path
import wandb

import math

from dataclasses import dataclass

import torch
from torch import autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F

from tqdm.auto import tqdm
import time
from get_args import get_args

from einops import rearrange
from safetensors.torch import save_file

from PIL import Image
from matplotlib import pyplot as plt

from diffusers import DiffusionPipeline, LMSDiscreteScheduler, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from diffusers.training_utils import cast_training_params

from transformers import CLIPProcessor, CLIPModel

from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers.training_utils import EMAModel

from peft import LoraConfig

args = get_args()
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
save_freq                       #Not used currently
max_grad_norm                   
accumulation_steps              
ema                             #Not used currently
save_dir                        #Not used currently

wandb
eval_prompts
eval_freq

test_loss_scale
"""

@dataclass
class Config:
    model_id = args.model_id

    target_prompt=args.target_prompt
    anchor_prompt=args.anchor_prompt
    time_max = args.time_max
    time_min = args.time_min
    infer_steps = args.infer_steps
    cfg = args.cfg
    guidance_scale = args.guidance_scale

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    mixed_precision = args.mixed_precision
    use_lora = args.use_lora
    lora_rank = args.lora_rank
    seed = args.seed
    lr = args.lr
    save_freq = args.save_freq
    max_grad_norm = args.max_grad_norm
    accumulation_steps = args.accumulation_steps
    ema = args.ema
    save_dir = args.save_dir

    wandb= args.wandb
    eval_prompts = args.eval_prompts
    eval_prompts = eval_prompts.split("_")
    eval_freq = args.eval_freq

    test_loss_scale = args.test_loss_scale

config = Config()

print(f"""
Configuration Summary:
----------------------
model_id        : {config.model_id}

target_prompt   : {config.target_prompt}
anchor_prompt   : {config.anchor_prompt}
time_max        : {config.time_max}
time_min        : {config.time_min}
infer_steps     : {config.infer_steps}
cfg             : {config.cfg}
guidance_scale  : {config.guidance_scale}

num_epochs      : {config.num_epochs}
batch_size      : {config.batch_size}
mixed_precision : {config.mixed_precision}
use_lora        : {config.use_lora}
lora_rank       : {config.lora_rank}
seed            : {config.seed}
lr              : {config.lr}
save_freq       : {config.save_freq}
max_grad_norm   : {config.max_grad_norm}

ema             : {config.ema}                  
save_dir        : {config.save_dir}

wandb           : {config.wandb}
eval_prompts    :  {config.eval_prompts}

test_loss_scale : {config.test_loss_scale}
""")

def set_seed(seed: int):
    import random, os
    import numpy as np
    import torch

    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

set_seed(config.seed)
save_dir = Path(config.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

import torch

def get_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    elif mixed_precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unknown mixed precision: {mixed_precision}")

torch_dtype = get_dtype(config.mixed_precision)

pipe = DiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch_dtype, safety_checker=None)
pipe_orig = DiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch_dtype, safety_checker=None)

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
unet = pipe.unet
unet_orig = pipe_orig.unet
vae = pipe.vae

scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(config.infer_steps)

for p in text_encoder.parameters():
    p.requires_grad = False

for p in vae.parameters():
    p.requires_grad = False

for p in unet_orig.parameters():
    p.requires_grad = False

for p in unet.parameters():
    p.requires_grad = False

if config.use_lora:
    unet_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if config.mixed_precision in ["fp16", "bf16"]:
        cast_training_params(unet, dtype=torch.float32)
else:
    for name, module in unet.named_modules():
        if (("transformer_blocks" in name)) and (("attn2" in name)):
            for n, p in module.named_parameters():
                p.requires_grad = True
        else:
            for n, p in module.named_parameters():
                p.requires_grad = False

optimizer = torch.optim.AdamW(unet.parameters(), lr=config.lr)
lr_scheduler = get_constant_schedule(
    optimizer=optimizer,
)

def generate(
        unet,
        vae,
        scheduler,
        cfg,
        guidance_scale,
        uncond_embeddings,
        text_embeddings,
        num_inference_steps = config.infer_steps,
        start_latents = None,
        return_traj=False,
):
    batch_size = uncond_embeddings.shape[0]
    height = 512                                # default height of Stable Diffusion
    width = 512                                 # default width of Stable Diffusion

    if scheduler is None:
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps)

    # Prepare latents
    if start_latents is None:
        latents = torch.randn(
        (batch_size, 4, height // 8, width // 8),
        )
    else:
        latents = start_latents

    latents = latents.to(uncond_embeddings.device)
    latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

    cat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    traj = []

    with autocast("cuda", dtype=torch_dtype):
        if cfg:
            for i, t in enumerate(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                with torch.inference_mode():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=cat_text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample
                if return_traj:
                    traj.append(latents)
        else:
            for i, t in enumerate(scheduler.timesteps):
                latent_model_input = latents.unsqueeze(0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                with torch.inference_mode():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                latents = scheduler.step(noise_pred, t, latents).prev_sample
                if return_traj:
                    traj.append(latents)
                    
    if vae is not None:
        latents = 1 / 0.18215 * latents
        with autocast("cuda"):
            with torch.inference_mode():
                image = vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    elif return_traj:
        return torch.stack(traj)
    else:
        return latents

def train_loop(config, unet, unet_orig, vae, text_encoder, tokenizer, optimizer, lr_scheduler, scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.accumulation_steps,
        log_with="wandb" if len(config.wandb) != 0 else None,
    )

    if len(config.wandb) != 0:
        accelerator.init_trackers(
            project_name=config.wandb
        )

    unet, unet_orig, vae, text_encoder, tokenizer, optimizer, lr_scheduler, scheduler = accelerator.prepare(
        unet, unet_orig, vae, text_encoder, tokenizer, optimizer, lr_scheduler, scheduler
    )

    unet.train()
    uncond_tokenized = tokenizer(
        [""] * config.batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(accelerator.device)
    uncond_embeds = text_encoder(uncond_tokenized)["last_hidden_state"]

    eval_uncond_embeds = uncond_embeds[0].repeat(len(config.eval_prompts), 1, 1)

    eval_prompts = config.eval_prompts
    eval_tokenized = tokenizer(
            eval_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(accelerator.device)
    eval_embeds = text_encoder(eval_tokenized)["last_hidden_state"]

    target_prompts = [config.target_prompt] * config.batch_size
    target_tokenized = tokenizer(
            target_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(accelerator.device)
    target_embeds = text_encoder(target_tokenized)["last_hidden_state"]

    anchor_prompts = [config.anchor_prompt] * config.batch_size
    anchor_tokenized = tokenizer(
            anchor_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(accelerator.device)
    anchor_embeds = text_encoder(anchor_tokenized)["last_hidden_state"]

    eval_fixed_latents = torch.randn(
        (eval_embeds.shape[0], 4, 64, 64),
        device=accelerator.device
    )

    # Prepare time indices
    scheduler.timesteps = scheduler.timesteps.to(accelerator.device)
    alphas = scheduler.alphas.to(accelerator.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device)

    t = scheduler.timesteps
    assert 1000 % config.infer_steps == 0
    shift = 1000//config.infer_steps

    idx_ts = (t >= config.time_min + shift) & (t <= config.time_max)
    idx_t_1s = (t >= config.time_min) & (t <= config.time_max - shift)

    idx_ts = torch.nonzero(idx_ts)
    idx_t_1s = torch.nonzero(idx_t_1s)
    
    # Sanity Check
    assert len(idx_ts) == len(idx_t_1s)
    global_step = 0

    if config.test_loss_scale:
        traj_batch = generate(
            unet=unet_orig,
            vae=None,
            scheduler=scheduler,
            cfg=config.cfg,
            guidance_scale=config.guidance_scale,
            uncond_embeddings=uncond_embeds,
            text_embeddings=anchor_embeds,
            num_inference_steps=config.infer_steps,
            start_latents=None,
            return_traj=True,
        )

        _, B, C, H, W = traj_batch.shape

        for idx_t, idx_t_1 in zip(idx_ts, idx_t_1s):
            
            t = scheduler.timesteps[idx_t].repeat(B)
            t_1 = scheduler.timesteps[idx_t_1].repeat(B)

            x_t = traj_batch[idx_t][0]
            x_t_1 = traj_batch[idx_t_1][0]

            noise_pred_x_t = unet_orig(x_t, t, encoder_hidden_states=anchor_embeds).sample

            with torch.inference_mode():
                noise_pred_x_t_1 = unet_orig(x_t_1, t_1, encoder_hidden_states=anchor_embeds).sample
            
            alpha_cumprod_t = alphas_cumprod[t].view(B, 1, 1, 1)
            alpha_cumprod_t_1 = alphas_cumprod[t_1].view(B, 1, 1, 1)

            # Wrong Scaling HERE
            # term1 = (x_t - torch.sqrt(alpha_t) * x_t_1)/torch.sqrt(1-alpha_cumprod_t)
            # term2 = (torch.sqrt(alpha_t)*torch.sqrt(1-alpha_cumprod_t_1)/torch.sqrt(1-alpha_cumprod_t))*noise_pred_x_t_1
            # target = term1 + term2

            target = (torch.sqrt(1-alpha_cumprod_t)*torch.sqrt(alpha_cumprod_t_1))/(torch.sqrt(alpha_cumprod_t)*torch.sqrt(1-alpha_cumprod_t_1))*noise_pred_x_t_1

            loss = F.mse_loss(noise_pred_x_t, target)
            logs = {"loss": loss.item()}
            accelerator.log(logs, step=global_step)
            global_step+=1
        return

    progress_bar = tqdm(range(config.num_epochs * config.batch_size), disable=not accelerator.is_local_main_process)
    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch {epoch}")

        if global_step % config.eval_freq == 0:
            eval_images = generate(
                unet=unet,
                vae=vae,
                scheduler=None,
                cfg=config.cfg,
                guidance_scale=config.guidance_scale,
                uncond_embeddings=eval_uncond_embeds,
                text_embeddings=eval_embeds,
                num_inference_steps=config.infer_steps,
                start_latents=eval_fixed_latents,
                return_traj=False,
            )

            for l, _ in enumerate(config.eval_prompts):
                accelerator.log({f"{config.eval_prompts[l]} {l}": [wandb.Image(eval_images[l], caption=f"step {global_step}")]}, step=global_step)

        optimizer.zero_grad()

        traj_batch = generate(
            unet=unet,
            vae=None,
            scheduler=scheduler,
            cfg=config.cfg,
            guidance_scale=config.guidance_scale,
            uncond_embeddings=uncond_embeds,
            text_embeddings=target_embeds,
            num_inference_steps=config.infer_steps,
            start_latents=None,
            return_traj=True,
        )

        _, B, C, H, W = traj_batch.shape

        for idx_t, idx_t_1 in zip(idx_ts, idx_t_1s):
            
            t = scheduler.timesteps[idx_t].repeat(B)
            t_1 = scheduler.timesteps[idx_t_1].repeat(B)

            x_t = traj_batch[idx_t][0]
            x_t_1 = traj_batch[idx_t_1][0]

            with accelerator.accumulate(unet):
                noise_pred_x_t = unet(x_t, t, encoder_hidden_states=target_embeds).sample

                with torch.inference_mode():
                    noise_pred_x_t_1 = unet_orig(x_t_1, t_1, encoder_hidden_states=anchor_embeds).sample

                alpha_t = alphas[t].view(B, 1, 1, 1)
                alpha_cumprod_t = alphas_cumprod[t].view(B, 1, 1, 1)
                alpha_cumprod_t_1 = alphas_cumprod[t_1].view(B, 1, 1, 1)

                # term1 = (x_t - torch.sqrt(alpha_t) * x_t_1)/torch.sqrt(1-alpha_cumprod_t)
                # term2 = (torch.sqrt(alpha_t)*torch.sqrt(1-alpha_cumprod_t_1)/torch.sqrt(1-alpha_cumprod_t))*noise_pred_x_t_1
                # target = term1 + term2

                target = (torch.sqrt(1-alpha_cumprod_t))/(torch.sqrt(alpha_t)*torch.sqrt(1-alpha_cumprod_t_1))*noise_pred_x_t_1

                loss = F.mse_loss(noise_pred_x_t, target)
                accelerator.backward(loss)
                
                logs = {"loss": loss.item()}

                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    logs["grad_norm"] = total_norm.item()
                
                optimizer.step()
                lr_scheduler.step()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
            
            logs["L2"] = x_t.mean().item()
        
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_step += 1

train_loop(config, unet, unet_orig, vae, text_encoder, tokenizer, optimizer, lr_scheduler, scheduler)