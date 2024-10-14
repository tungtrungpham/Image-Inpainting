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

from segment_anything import build_sam_vit_b, SamPredictor
from lama_inpaint.pipeline import ImageInpaintingPipeline
from stable_diffusion_inpaint.sd_inpaint import fill_img_with_sd, replace_img_with_sd


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


def dilate_mask(mask, dilate_factor=15):
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


def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

 
def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w):
    point_coords = [w, h]
    point_labels = [1]
    dilate_kernel_size = 15

    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    
    point_input = np.array([point_coords])
    point_labels_input = np.array(point_labels)
    
    masks, _, _ = model['sam'].predict(
        point_coords=point_input, #np.array([point_coords]),
        point_labels=point_labels_input,#np.array(point_labels),
        multimask_output=True,
    )

    masks = masks.astype(np.uint8) * 255
    masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

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
            size=(width*0.04)**2)
        show_mask(
            plt.gca(),
            mask,
            random_color=False)
        plt.tight_layout()
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    return *figs, *masks

     
def get_inpainted_img(img, mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    input = {'img': img, 'mask': mask}
    img_inpainted = model['lama'].preprocess(input)
    return img_inpainted


def get_replaced_img(img, mask, text_prompt, negative_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    np_img = np.array(img, dtype=np.uint8)
    #H, W, C = np_img.shape # Is this redundant? 
    #np_img = HWC3(np_img)

    img_replaced = replace_img_with_sd(np_img, mask, text_prompt, negative_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced


def get_filled_img(img, mask, text_prompt, negative_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    np_img = np.array(img, dtype=np.uint8)
    #H, W, C = np_img.shape # Is this redundant? 
    #np_img = HWC3(np_img)

    filled_img = fill_img_with_sd(np_img, mask, text_prompt, negative_prompt, device=device)
    filled_img = filled_img.astype(np.uint8)
    return filled_img


# build model

# --- build lama -------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = {}
model['lama'] = ImageInpaintingPipeline(model_dir='./pretrained_weights/pytorch_model.pt', deivce=device)

# --- build sam -------
sam = build_sam_vit_b('./pretrained_weights/sam_vit_b_01ec64.pth')
sam.to(device)
model['sam'] = SamPredictor(sam)


button_size = (100,50)
with gr.Blocks(theme='shivi/calm_seafoam') as demo:
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

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
            mask_choice = gr.Dropdown(choices=["Mask 0", "Mask 1", "Mask 2"], label="Select Mask Meets Your Needs").style(full_width=True, size="sm")
            text_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe new object or background")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Describe what you don't want in the image")
            clear_button_image = gr.Button(value="Reset", label="Reset", icon="reset", variant="secondary").style(full_width=True, size="sm")


    mask_0 = gr.Image(type="numpy", label="Mask 0", visible=False)
    mask_1 = gr.Image(type="numpy", label="Mask 1", visible=False)
    mask_2 = gr.Image(type="numpy", label="Mask 2", visible=False)

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("### Masked Image")
            with gr.Row():
                img_with_mask_0 = gr.Plot(label="Mask 0")
                img_with_mask_1 = gr.Plot(label="Mask 1")
                img_with_mask_2 = gr.Plot(label="Mask 2")

    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Markdown("### Image Removed Object")
            img_rm_with_mask = gr.outputs.Image(
                type="numpy", label="Image Removed Object").style(height="200px")
        
        with gr.Column():
            gr.Markdown("### Image Replaced Background")
            img_replace_with_mask = gr.outputs.Image(
                type="numpy", label="Image Replaced Object").style(height="200px")

        with gr.Column():
            gr.Markdown("### Image Replaced Object")
            img_fill_with_mask = gr.outputs.Image(
                type="numpy", label="Image Replaced Background").style(height="200px")


    def get_select_coords(img, evt: gr.SelectData):
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        show_points(plt.gca(), [[evt.index[0], evt.index[1]]], [1],
                    size=(width*0.04)**2)
        return evt.index[0], evt.index[1], fig

    img.select(get_select_coords, [img], [w, h, img_pointed])
    img.upload(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])

    sam_mask.click(
        get_masked_img,
        [img, w, h, features, orig_h, orig_w, input_h, input_w],
        [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
    )

    lama.click(
        lambda img, mask_choice, mask_0, mask_1, mask_2: get_inpainted_img(
            img, 
            {"Mask 0": mask_0, "Mask 1": mask_1, "Mask 2": mask_2}[mask_choice]),
        [img, mask_choice, mask_0, mask_1, mask_2],
        img_rm_with_mask
    )

    replace.click(
        lambda img, mask_choice, mask_0, mask_1, mask_2, text_prompt, negative_prompt: get_replaced_img(
            img,
            {"Mask 0": mask_0, "Mask 1": mask_1, "Mask 2": mask_2}[mask_choice],
            text_prompt, 
            negative_prompt),
        [img, mask_choice, mask_0, mask_1, mask_2, text_prompt, negative_prompt],
        img_replace_with_mask
    )

    fill.click(
        lambda img, mask_choice, mask_0, mask_1, mask_2, text_prompt, negative_prompt: get_filled_img(
            img,
            {"Mask 0": mask_0, "Mask 1": mask_1, "Mask 2": mask_2}[mask_choice],
            text_prompt, 
            negative_prompt),
        [img, mask_choice, mask_0, mask_1, mask_2, text_prompt, negative_prompt],
        img_fill_with_mask
    )

    def reset(*args):
        return [None for _ in args]
    
    clear_button_image.click(
        reset,
        [img, img_pointed, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask],
        [img, img_pointed, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask]
    )

if __name__ == "__main__":
    demo.launch(share=False)