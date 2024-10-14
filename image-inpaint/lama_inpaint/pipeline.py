from typing import Dict
import cv2
import numpy as np
import requests
from io import BytesIO
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from .build_lama import FFTInpainting


class ImageInpaintingPipeline():
    def __init__(self, model_dir, device=None, pad_out_to_modulo=8, refine=False, **kwargs):
        self.refine = refine
        self.pad_out_to_modulo = pad_out_to_modulo
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = FFTInpainting(model_dir=model_dir)
        self.infer_model.eval()
        self.infer_model = self.move_to_device(self.infer_model, self.device)
        
    def move_to_device(self, obj, device):
        if isinstance(obj, nn.Module):
            return obj.to(device)
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return [self.move_to_device(el, device) for el in obj]
        if isinstance(obj, dict):
            return {
                name: self.move_to_device(val, device)
                for name, val in obj.items()
            }
        raise ValueError(f'Unexpected type {type(obj)}')
        
    def transforms(self, img):
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        return out_img

    def ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def pad_img_to_modulo(self, img, mod):
        if len(img.shape) > 3:
            img = img.squeeze()
        channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return np.pad(
            img, ((0, 0), (0, out_height - height), (0, out_width - width)),
            mode='symmetric')
    
    def preprocess(self, input: Dict):
        img = input['img']
        img = self.transforms(img)
        mask = input['mask']
        mask = self.transforms(mask)
            
        
        result = dict(
            unpad_to_size=img.shape[1:],
            image=self.pad_img_to_modulo(img, self.pad_out_to_modulo),
            mask=self.pad_img_to_modulo(np.expand_dims(mask, axis=0), self.pad_out_to_modulo))
        
        result = self.perform_inference(result)
        return result
    
    def perform_inference(self, data):
        #batch = default_collate([data])
        batch = self.move_to_device(default_collate([data]), self.device)
        if self.refine:
            assert 'unpad_to_size' in batch, 'Unpadded size is required for the refinement'
            assert 'cuda' in str(self.device), 'GPU required for refinement'
            gpu_ids = str(self.device).split(':')[-1]
            cur_res = refine_predict(
                batch,
                self.infer_model,
                gpu_ids=gpu_ids,
                modulo=self.pad_out_to_modulo,
                n_iters=15,
                lr=0.002,
                min_side=512,
                max_scales=3,
                px_budget=900000)
            cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = self.infer_model(batch)
                cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        return cur_res

