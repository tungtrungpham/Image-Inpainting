import gradio as gr
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import tempfile
from PIL import Image
from typing import Any, Dict, List
from pathlib import Path

from lama_inpaint.pipeline import ImageInpaintingPipeline
from stable_diffusion_inpaint.sd_inpaint import fill_img_with_sd, replace_img_with_sd
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers import StableDiffusionInpaintPipeline


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)


def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


# Fill holes and smooth edges of mask
def dilate_mask(mask, dilate_factor):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)

# Set image right after upload for accelerating generate mask proces
def get_sam_feat(img):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        model['sam'].set_image(img)
        features = model['sam']._features 
        orig_hw = model['sam']._orig_hw
        return features, orig_hw

 
def get_masked_img(img, w, h, features, orig_hw):
    point_coords = [w, h]
    point_labels = [1]

    model['sam']._is_image_set = True
    model['sam']._features = features
    model['sam']._orig_hw = orig_hw

    point_input = np.array([point_coords])
    point_labels_input = np.array(point_labels)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
        masks, _, _ = model['sam'].predict(
            point_coords=point_input, 
            point_labels=point_labels_input,
            multimask_output=True,
        )
    masks = masks.astype(np.uint8) * 255
    figs = []
    for idx, mask in enumerate(masks):
        tmp_p = mkstemp('.png')
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(
            plt.gca(), 
            [point_coords], 
            point_labels, 
            size=(width*0.04)**2
        )
        show_mask(
            plt.gca(),
            mask,
            random_color=False
        )
        plt.tight_layout()
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    return *figs, *masks


def get_inpainted_img(img, mask):
    mask = dilate_mask(mask, dilate_factor=15)
    input = {'img': img, 'mask': mask}
    img_inpainted = model['lama'].preprocess(input)
    return img_inpainted


def get_replaced_img(img, mask, text_prompt, negative_prompt):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    np_img = np.array(img, dtype=np.uint8)

    img_replaced = replace_img_with_sd(
        pipe=model['sd_pipe'],
        img=np_img, mask=mask,
        text_prompt=text_prompt,
        negative_prompt=negative_prompt
    )
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced

def get_filled_img(img, mask, text_prompt, negative_prompt):
    mask = dilate_mask(mask, dilate_factor=5)
    np_img = np.array(img, dtype=np.uint8)

    filled_img = fill_img_with_sd(
        pipe=model['sd_pipe'],
        img=np_img,
        mask=mask,
        text_prompt=text_prompt,
        negative_prompt=negative_prompt
    )
    filled_img = filled_img.astype(np.uint8)
    return filled_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = {}

# ---- build LaMA ----
model['lama'] = ImageInpaintingPipeline(model_dir='./pretrained_weights/pytorch_model.pt', deivce=device)

# ---- build SAM ----
model['sam'] = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# ---- build SD ----
model['sd_pipe'] = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,).to(device)


button_size = (50,25)
with gr.Blocks(theme='shivi/calm_seafoam') as demo:
    features = gr.State(None)
    orig_hw = gr.State(None)

    with gr.Row().style(mobile_collapse=False, equal_height=True):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("### Input Image")
            with gr.Row():
                img = gr.Image(label="Input Image").style(height="200px")
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("### Pointed Image")
            with gr.Row():
                img_pointed = gr.Plot(label='Pointed Image')
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("### Control Panel")
           
            w = gr.Number(label="Point Coordinate W", visible=False)
            h = gr.Number(label="Point Coordinate H", visible=False)

            sam_mask = gr.Button("Predict Mask Using SAM", variant="primary").style(full_width=True, size="sm")
            lama = gr.Button("Remove Masked Object", variant="primary").style(full_width=True, size="sm")
            replace = gr.Button("Replace Backgound", variant="primary").style(full_width=True, size="sm")
            fill = gr.Button("Replace Masked Object", variant="primary").style(full_width=True, size="sm")
            mask_choice = gr.Dropdown(choices=["Mask 1", "Mask 2", "Mask 3"], label="Select Mask Meets Your Needs").style(full_width=True, size="sm")
            text_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe new object or background")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Describe what you don't want in the image")
            clear_button_image = gr.Button(value="Reset", label="Reset", icon="reset", variant="secondary").style(full_width=True, size="sm")


    mask_1 = gr.Image(type="numpy", label="Mask 1", visible=False)
    mask_2 = gr.Image(type="numpy", label="Mask 2", visible=False)
    mask_3 = gr.Image(type="numpy", label="Mask 3", visible=False)

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("### Masked Image")
            with gr.Row():
                img_with_mask_1 = gr.Plot(label="Mask 1")
                img_with_mask_2 = gr.Plot(label="Mask 2")
                img_with_mask_3 = gr.Plot(label="Mask 3")

    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Markdown("### Image Removed Object")
            img_removed = gr.outputs.Image(type="numpy", label="Image Removed Object").style(height="200px")
        
        with gr.Column():
            gr.Markdown("### Image Replaced Background")
            img_replaced = gr.outputs.Image(type="numpy", label="Image Replaced Background").style(height="200px")

        with gr.Column():
            gr.Markdown("### Image Replaced Object")
            img_filled = gr.outputs.Image(type="numpy", label="Image Replaced Object").style(height="200px")


    def get_select_coords(img, evt: gr.SelectData):
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        show_points(
            plt.gca(),
            [[evt.index[0], evt.index[1]]], 
            [1],
            size=(width*0.04)**2
        )
        return evt.index[0], evt.index[1], fig

    img.select(get_select_coords, [img], [w, h, img_pointed])
    img.upload(get_sam_feat, [img], [features, orig_hw])

    sam_mask.click(
        get_masked_img,
        [img, w, h, features, orig_hw],
        [img_with_mask_1, img_with_mask_2, img_with_mask_3, mask_1, mask_2, mask_3]
    )

    lama.click(
        lambda img, mask_choice, mask_1, mask_2, mask_3: get_inpainted_img(
            img, 
            {"Mask 1": mask_1, "Mask 2": mask_2, "Mask 3": mask_3}[mask_choice]
        ),
        [img, mask_choice, mask_1, mask_2, mask_3],
        img_removed
    )

    replace.click(
        lambda img, mask_choice, mask_1, mask_2, mask_3, text_prompt, negative_prompt: get_replaced_img(
            img,
            {"Mask 1": mask_1, "Mask 2": mask_2, "Mask 3": mask_3}[mask_choice],
            text_prompt, 
            negative_prompt
        ),
        [img, mask_choice, mask_1, mask_2, mask_3, text_prompt, negative_prompt],
        img_replaced
    )

    fill.click(
        lambda img, mask_choice, mask_1, mask_2, mask_3, text_prompt, negative_prompt: get_filled_img(
            img,
            {"Mask 1": mask_1, "Mask 2": mask_2, "Mask 3": mask_3}[mask_choice],
            text_prompt, 
            negative_prompt
        ),
        [img, mask_choice, mask_1, mask_2, mask_3, text_prompt, negative_prompt],
        img_filled
    )

    def reset(*args):
        return [None for _ in args]
    
    clear_button_image.click(
        reset,
        [img, img_pointed, img_with_mask_1, img_with_mask_2, img_with_mask_3, img_removed, img_filled, img_replaced],
        [img, img_pointed, img_with_mask_1, img_with_mask_2, img_with_mask_3, img_removed, img_filled, img_replaced]
    )

if __name__ == "__main__":
    demo.launch(share=True)
