"""
Most of the code is borrowed and modified from LaMa, available here https://github.com/saic-mdal/lama
"""
import torch
import torch.nn as nn

from .generator import FFCResNetGenerator


class DefaultInpaintingTrainingModule(nn.Module):
    def __init__(self, model_dir=''):
        super().__init__()
        self.generator = FFCResNetGenerator()
        state = torch.load(model_dir, weights_only=True, map_location='cpu')
        self.load_state_dict(state, strict=False)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * img
        return batch


class FFTInpainting(nn.Module):
    def __init__(self, model_dir=''):
        super(FFTInpainting, self).__init__()
        self.model = DefaultInpaintingTrainingModule(model_dir=model_dir)

    def forward(self, inputs):
        return self.model(inputs)

