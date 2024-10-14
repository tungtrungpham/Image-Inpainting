from typing import Optional
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from .utils import crop_for_fill_pred, crop_for_fill_post, resize_and_pad, recover_size

def fill_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        negative_prompt: Optional[str] = None,
        device="cuda"
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    img_crop_filled = pipe(
        prompt=text_prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop)
    ).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled


def replace_img_with_sd(
        img: np.ndarray,
        mask: np.ndarray,
        text_prompt: str,
        negative_prompt: Optional[str] = None,
        step: int = 50,
        device="cuda"
):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = pipe(
        prompt=text_prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(img_padded),
        mask_image=Image.fromarray(255 - mask_padded),
        num_inference_steps=step,
    ).images[0]
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1-mask_resized) + img * mask_resized
    return img_resized