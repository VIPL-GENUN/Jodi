# This file is modified from https://github.com/NVlabs/Sana

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.abspath('.'))

import datetime
import hashlib
import itertools
import os.path as osp
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from PIL import Image
from termcolor import colored

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS, Scheduler
from data.builder import build_dataloader, build_dataset
from data.datasets.jodi_dataset import RandomConcatJodiDataset
from data.sampler import DistributedRangedSampler
from model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.respace import compute_density_for_timestep_sampling
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import JodiConfig
from utils.data_sampler import AspectRatioBatchSampler
from utils.dist_utils import flush, get_world_size
from utils.logger import LogBuffer, get_root_logger
from utils.lr_scheduler import build_lr_scheduler
from utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from utils.optimizer import auto_scale_lr, build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)

    logger.info("Running validation... ")

    # Run sampling
    latents = []
    for prompt in validation_prompts:
        z = torch.randn(1, 1 + num_conditions, config.vae.vae_latent_dim, latent_size, latent_size, device=device)
        embed = torch.load(
            osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
            map_location="cpu",
        )
        caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
        role = torch.zeros((1, 1 + num_conditions), dtype=torch.long, device=device)
        model_kwargs = dict(mask=emb_masks, role=role)
        dpm_solver = DPMS(
            model.forward,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=4.5,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        denoised = dpm_solver.sample(
            z,
            steps=20,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=config.scheduler.flow_shift,
        )
        latents.append(denoised)
    torch.cuda.empty_cache()

    # Decode latents
    image_logs = []
    for prompt, latent in zip(validation_prompts, latents):
        latent = latent.to(torch.float16)
        latent = torch.unbind(latent, dim=1)
        images = []
        for lat in latent:
            sample = vae_decode(config.vae.vae_type, vae, lat)
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
            sample = sample.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            images.append(Image.fromarray(sample))
        image_logs.append({"validation_prompt": prompt, "images": images})

    # Save images
    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io
        images = list(itertools.chain.from_iterable([log["images"] for log in image_caption]))
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]
        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))
        new_im = Image.new("RGB", (max_width * images_per_row, total_height))
        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)
        return new_im

    file_format = "png"  # "webp"
    local_vis_save_path = osp.join(config.work_dir, "log_vis")
    os.umask(0o000)
    os.makedirs(local_vis_save_path, exist_ok=True)
    concatenated_image = concatenate_images(image_logs, images_per_row=num_conditions+1, image_format=file_format)
    save_path = osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
    concatenated_image.save(save_path)

    model.train()
    flush()
    return image_logs


def train(config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = train_dataloader.batch_sampler.sampler
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # NOTE: train_dataloader is infinite since batch_sampler is infinite
            # therefore, we actually stay in this for loop until the end of training
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=(config.model.mixed_precision == "fp16" or config.model.mixed_precision == "bf16"),
                ):
                    z = []
                    imgs = torch.unbind(batch["img"], dim=1)
                    for img in imgs:
                        z.append(vae_encode(
                            config.vae.vae_type, vae, img, config.vae.sample_posterior, accelerator.device
                        ))
                    z = torch.stack(z, dim=1)

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            clean_images = z

            lm_time_start = time.time()
            if "T5" in config.text_encoder.text_encoder_name:
                with torch.no_grad():
                    txt_tokens = tokenizer(
                        batch["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]
            elif (
                "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
            ):
                with torch.no_grad():
                    chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                    prompt = [chi_prompt + i for i in batch["text"]]
                    num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                    max_length_all = (
                        num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                    )  # magic number 2: [bos], [_]
                    txt_tokens = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=max_length_all,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(
                        range(-config.text_encoder.model_max_length + 1, 0)
                    )  # first bos and end N-1
                    y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                        :, :, select_index
                    ]
                    y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
            else:
                print("error")
                exit()

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_images.device)

            # Get the role
            role = batch["role"]
            assert role.shape == (bs, 1+num_conditions)

            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model, clean_images, timesteps,
                    model_kwargs=dict(y=y, role=role, mask=y_mask, clean_x=clean_images),
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.training_hours
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )

                if (time.time() - training_start_time) / 3600 > config.train.training_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return
            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(
                        accelerator=accelerator,
                        config=config,
                        model=model,
                        logger=logger,
                        step=global_step,
                        device=accelerator.device,
                    )

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

        accelerator.wait_for_everyone()


@pyrallis.wrap()
def main(cfg: JodiConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global text_encoder, tokenizer
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps
    global num_conditions

    config = cfg
    args = cfg

    training_start_time = time.time()

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    # Initialize accelerator
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )

    # Initialize tensorboard logging
    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    # Set random seed
    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb
            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    # logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}")
    logger.info(f"seed: {config.train.seed}")
    logger.info(f"Initializing: DDP for training")

    # Get pretrained VAE
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    logger.info(f"vae type: {config.vae.vae_type}")

    # Get tokenizer and text encoder
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, accelerator.device)
    logger.info(f"text encoder: {config.text_encoder.text_encoder_name}")

    # Compute and save null embedding and validation prompts embeddings
    os.makedirs(config.train.null_embed_root, exist_ok=True)
    text_embed_dim = text_encoder.config.hidden_size
    chi_prompt = "\n".join(config.text_encoder.chi_prompt)
    # logger.info(f"Complex Human Instruct: {chi_prompt}")
    max_length = config.text_encoder.model_max_length
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    if config.train.visualize and len(config.train.validation_prompts):
        # Preparing embeddings for visualization. We put it here for saving GPU memory
        valid_prompt_embed_suffix = f"{max_length}token_{config.text_encoder.text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        skip = True
        uuid_chi_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_chi_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        # Save complex human instruct to a file
        chi_prompt_file = osp.join(config.train.valid_prompt_embed_root, "chi_prompt.txt")
        with open(chi_prompt_file, "w", encoding="utf-8") as f:
            f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            for prompt in validation_prompts:
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if "T5" in config.text_encoder.text_encoder_name:
                    txt_tokens = tokenizer(
                        prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                    caption_emb_mask = txt_tokens.attention_mask
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                    prompt = chi_prompt + prompt
                    num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                    max_length_all = (
                        num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                    )  # magic number 2: [bos], [_]

                    txt_tokens = tokenizer(
                        prompt,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][
                        :, select_index
                    ]
                    caption_emb_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                torch.save({"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}, prompt_embed_path)

            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            if "T5" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            elif "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_tokens.attention_mask},
                null_embed_path,
            )
            del null_token_emb
            del null_tokens
            flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # Build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    logger.info(f"v-prediction: {config.scheduler.predict_v}")
    logger.info(f"noise schedule: {config.scheduler.noise_schedule}")
    if "flow" in config.scheduler.noise_schedule:
        logger.info(f"flow shift: {config.scheduler.flow_shift}")
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        logger.info(f"flow weighting: {config.scheduler.weighting_scheme}")
        logger.info(f"logit-mean: {config.scheduler.logit_mean}")
        logger.info(f"logit-std: {config.scheduler.logit_std}")

    # Build models
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    num_conditions = len(config.data.datasets[0]["conditions"])
    logger.info(f"Number of conditions: {num_conditions}")
    model_kwargs = {
        "config": config,
        "in_channels": config.vae.vae_latent_dim,
        "mlp_ratio": config.model.mlp_ratio,
        "caption_channels": text_embed_dim,
        "pe_interpolation": config.model.pe_interpolation,
        "model_max_length": max_length,
        "qk_norm": config.model.qk_norm,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "mlp_acts": list(config.model.mlp_acts),
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "num_conditions": num_conditions,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).train()
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    # Load weights
    load_from = True
    state_dict = None
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )
    if args.load_from is not None:
        config.model.load_from = args.load_from  # type: ignore
    if config.model.load_from is not None and load_from:
        state_dict, _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    # Build dataset
    dataset = [build_dataset(d) for d in config.data.datasets]
    dataset = RandomConcatJodiDataset(dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    accelerator.wait_for_everyone()

    # Build dataloader
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True)
    batch_sampler = AspectRatioBatchSampler(
        sampler=sampler,
        dataset=dataset,
        batch_size=config.train.train_batch_size,
        aspect_ratios=dataset.aspect_ratio,
        ratio_nums=dataset.ratio_nums,
        drop_last=True,
        config=config,
        caching=args.caching,
    )
    train_dataloader = build_dataloader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.train.num_workers,
    )
    train_dataloader_len = len(train_dataloader)

    # Build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.info(f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}")
    logger.info(f"lr: {config.train.optimizer['lr']:.5f}")
    logger.info(f"bs: {config.train.train_batch_size}")
    logger.info(f"gc: {config.train.grad_checkpointing}")
    logger.info(f"gc_accum_step: {config.train.gradient_accumulation_steps}")
    logger.info(f"qk norm: {config.model.qk_norm}")
    logger.info(f"fp32 attn: {config.model.fp32_attention}")
    logger.info(f"attn type: {config.model.attn_type}")
    logger.info(f"text encoder: {config.text_encoder.text_encoder_name}")
    logger.info(f"precision: {config.model.mixed_precision}")

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            ignore_keys = []
            if args.extend_to_new_domain:
                ignore_keys = ["domain_embedding"]
            state_dict, _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer if check_flag else None,
                lr_scheduler=lr_scheduler if check_flag else None,
                null_embed_path=null_embed_path,
                ignore_keys=ignore_keys,
            )
            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

        path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # resume randomise
        if rng_state:
            logger.info("resuming randomise")
            torch.set_rng_state(rng_state["torch"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            generator.set_state(rng_state["generator"])  # resume generator status
            try:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            except:
                logger.warning("Failed to resume torch_cuda rng state")

    if state_dict is not None:
        # resume from sana: copy x_embedder and final_layer
        if args.resume_from_sana:
            for x_embedder in model.x_embedders:
                x_embedder.proj.weight.data.copy_(state_dict["x_embedder.proj.weight"])
                x_embedder.proj.bias.data.copy_(state_dict["x_embedder.proj.bias"])
            model.final_layers[0].linear.weight.data.copy_(state_dict["final_layer.linear.weight"])
            model.final_layers[0].linear.bias.data.copy_(state_dict["final_layer.linear.bias"])
            model.final_layers[0].scale_shift_table.data.copy_(state_dict["final_layer.scale_shift_table"])
            logger.info("Copied x_embedder and final_layer weights from pretrained model.")
        # extend to new domain: copy x_embedder and final_layer from specified domain, copy domain embedding
        elif args.extend_to_new_domain:
            num_domains = state_dict["domain_embedding"].shape[0]
            model.domain_embedding[:num_domains].data.copy_(state_dict["domain_embedding"])
            idx = args.extend_to_new_domain_copy_id
            if not isinstance(idx, list):
                idx = [idx]
            for i in range(len(idx)):
                model.x_embedders[num_domains+i].proj.weight.data.copy_(state_dict[f"x_embedders.{idx[i]}.proj.weight"])
                model.x_embedders[num_domains+i].proj.bias.data.copy_(state_dict[f"x_embedders.{idx[i]}.proj.bias"])
                model.final_layers[num_domains+i].linear.weight.data.copy_(state_dict[f"final_layers.{idx[i]}.linear.weight"])
                model.final_layers[num_domains+i].linear.bias.data.copy_(state_dict[f"final_layers.{idx[i]}.linear.bias"])
                model.final_layers[num_domains+i].scale_shift_table.data.copy_(state_dict[f"final_layers.{idx[i]}.scale_shift_table"])
            logger.info("Copied x_embedder, final_layer and domain embedding weights from pretrained model.")
        # resume from previous checkpoint: state dicts must match
        else:
            assert len(missing) == 0, f"Missing keys: {missing}"  # noqa
            assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"  # noqa
            logger.info("Successfully loaded pretrained model.")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":
    main()
