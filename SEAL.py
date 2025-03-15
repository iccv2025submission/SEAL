from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import zlib
import wandb
import random
import os
from utils import *
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Noise patch detection analysis with enhanced brute-force simhash noise selection')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='noise-detection')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project=args.wandb_project, name="SEAL", entity=args.wandb_entity, config=vars(args))

    # Load models
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        revision='fp16',
    ).to(device)
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)
    sentence_model = SentenceTransformer("iccv2025submission/finetuned-caption-embedding").to(device)

    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)

    all_min_l2 = {}

    for k in args.k_values:
        for b in args.b_values:
            min_l2_wm = []
            min_l2_random = []
            
            for img_idx in tqdm(range(args.start, args.end), desc=f'k={k}, b={b}'):
                prompt_1 = dataset[img_idx]['Prompt']
                prompt_2 = dataset[(img_idx+1) % len(dataset)]['Prompt']
                
                # Generate watermarked image and noise
                first_image = pipe(prompt_1).images[0]
                image_caption = generate_caption(first_image, cap_processor, cap_model)
                embedding = sentence_model.encode(image_caption, convert_to_tensor=True).to(device)
                embedding = embedding / torch.norm(embedding)

                image_noise = generate_initial_noise(embedding, k, b, 42, device).to(dtype=pipe.vae.dtype)
                image = pipe(prompt_1, latents=image_noise).images[0]
                org_img = image

                # Generate noise again from the watermarked image's caption
                org_caption = generate_caption(org_img, cap_processor, cap_model)
                org_embedding = sentence_model.encode(org_caption, convert_to_tensor=True).to(device)
                org_embedding = org_embedding / torch.norm(org_embedding)

                org_noise = generate_initial_noise(org_embedding, k, b, 42, device).to(dtype=pipe.vae.dtype)

                # Inverse the watermarked image
                org_tensor = transform_img(org_img).unsqueeze(0).to(device)
                org_tensor = org_tensor.to(dtype=pipe.vae.dtype)
                org_latents = pipe.get_image_latents(org_tensor, sample=False)
                recon_noise = pipe.forward_diffusion(
                    latents=org_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )

                # Generate a random image and corresponding noise
                random_image = pipe(prompt_2).images[0]
                random_tensor = transform_img(random_image).unsqueeze(0).to(device)
                random_tensor = random_tensor.to(dtype=pipe.vae.dtype)
                random_latents = pipe.get_image_latents(random_tensor, sample=False)
                random_recon_noise = pipe.forward_diffusion(
                    latents=random_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )

                random_caption = generate_caption(random_image, cap_processor, cap_model)
                rand_embedding = sentence_model.encode(random_caption, convert_to_tensor=True).to(device)
                rand_embedding = rand_embedding / torch.norm(rand_embedding)

                random_noise = generate_initial_noise(rand_embedding, k, b, 42, device).to(dtype=pipe.vae.dtype)

                wm_l2 = calculate_patch_l2(recon_noise, org_noise, k)
                random_l2 = calculate_patch_l2(random_recon_noise, random_noise, k)
                min_l2_wm.append(wm_l2)
                min_l2_random.append(random_l2)

            # Store the results for this combination in the dictionary
            combo_key = f"{k}_{b}"
            all_min_l2[combo_key] = {"watermarked": np.array(min_l2_wm), "random": np.array(min_l2_random)}
    
    
    wandb.finish()
