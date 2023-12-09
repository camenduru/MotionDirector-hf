import os
import warnings
from typing import Optional

import torch
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
from utils.lora import extract_lora_child_module
import imageio


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = True,
    sdp: bool = True,
    lora_path: str = "",
    lora_rank: int = 32,
    lora_scale: float = 1.0,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(xformers, sdp, unet)

    lora_manager_temporal = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        use_text_lora=False,
        save_for_webui=False,
        only_for_webui=False,
        unet_replace_modules=["TransformerTemporalModel"],
        text_encoder_replace_modules=None,
        lora_bias=None
    )

    unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
        True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale)

    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe


def inverse_video(pipe, latents, num_steps):
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path:str,
    model_select: str,
    random_seed: int,
):
    # initialize with random gaussian noise
    scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
    if random_seed > 1000:
        torch.manual_seed(random_seed)
    else:
        random_seed = random.randint(100, 10000000)
        torch.manual_seed(random_seed)
        print(f"random_seed: {random_seed}")
    if '1-' in model_select:
        noise_prior = 0.3
    elif '2-' in model_select:
        noise_prior = 0.5
    elif '3-' in model_select:
        noise_prior = 0.
    else:
        noise_prior = 0.
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path)
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = torch.load(latents_path)['inversion_noise'].unsqueeze(0)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape != shape:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), (height // scale, width // scale), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        noise = torch.randn_like(latents, dtype=torch.half)
        latents_base = noise
        latents = (noise_prior) ** 0.5 * latents + (1 - noise_prior) ** 0.5 * noise
    else:
        latents = torch.randn(shape, dtype=torch.half)
        latents_base = latents

    return latents, latents_base, random_seed


class MotionDirector():
    def __init__(self):
        self.version = "0.0.0"
        self.foundation_model_path = "./zeroscope_v2_576w/"
        self.lora_path = "./MotionDirector_pretrained/dolly_zoom_(hitchcockian_zoom)/checkpoint-default/temporal/lora"
        with torch.autocast("cuda", dtype=torch.half):
            self.pipe = initialize_pipeline(model=self.foundation_model_path, lora_path=self.lora_path, lora_scale=1)

    def reload_lora(self, lora_path):
        if lora_path != self.lora_path:
            self.lora_path = lora_path
            with torch.autocast("cuda", dtype=torch.half):
                self.pipe = initialize_pipeline(model=self.foundation_model_path, lora_path=self.lora_path)

    def __call__(self, model_select, text_pormpt, neg_text_pormpt, random_seed, steps, guidance_scale, baseline_select):
        model_select = str(model_select)
        out_name = f"./outputs/inference"
        out_name += f"{text_pormpt}".replace(' ', '_').replace(',', '').replace('.', '')

        model_select_type = model_select.split('--')[1].strip()
        model_select_type = model_select_type.lower().replace(' ', '_')

        lora_path = f"./MotionDirector_pretrained/{model_select_type}/checkpoint-default/temporal/lora"
        self.reload_lora(lora_path)
        latents_folder = f"./MotionDirector_pretrained/{model_select_type}/cached_latents"
        latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(lora_path)

        device = "cuda"
        with torch.autocast(device, dtype=torch.half):
            # prepare input latents
            with torch.no_grad():
                init_latents, init_latents_base, random_seed = prepare_input_latents(
                    pipe=self.pipe,
                    batch_size=1,
                    num_frames=16,
                    height=384,
                    width=384,
                    latents_path=latents_path,
                    model_select=model_select,
                    random_seed=random_seed
                )
                video_frames = self.pipe(
                    prompt=text_pormpt,
                    negative_prompt=neg_text_pormpt,
                    width=384,
                    height=384,
                    num_frames=16,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    latents=init_latents
                ).frames

                out_file = f"{out_name}_{random_seed}.mp4"
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                export_to_video(video_frames, out_file, 8)

                if baseline_select:
                    with torch.autocast("cuda", dtype=torch.half):

                        loras = extract_lora_child_module(self.pipe.unet, target_replace_module=["TransformerTemporalModel"])
                        for lora_i in loras:
                            lora_i.scale = 0.

                        # self.pipe = initialize_pipeline(model=self.foundation_model_path, lora_path=self.lora_path,
                        #                                 lora_scale=0.)
                        with torch.no_grad():
                            video_frames = self.pipe(
                                prompt=text_pormpt,
                                negative_prompt=neg_text_pormpt,
                                width=384,
                                height=384,
                                num_frames=16,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                latents=init_latents_base,
                            ).frames

                            out_file_baseline = f"{out_name}_{random_seed}_baseline.mp4"
                            os.makedirs(os.path.dirname(out_file_baseline), exist_ok=True)
                            export_to_video(video_frames, out_file_baseline, 8)
                    # with torch.autocast("cuda", dtype=torch.half):
                    #     self.pipe = initialize_pipeline(model=self.foundation_model_path, lora_path=self.lora_path,
                    #                                     lora_scale=1.)
                    loras = extract_lora_child_module(self.pipe.unet,
                                                      target_replace_module=["TransformerTemporalModel"])
                    for lora_i in loras:
                        lora_i.scale = 1.

                else:
                    out_file_baseline = None

        return [out_file, out_file_baseline]
