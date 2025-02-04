import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class CLIP(nn.Module):
    
    def __init__(self, image_model, meta_model, normalize:bool = True):
        super().__init__()
        
        self.image_model = image_model
        self.meta_model = meta_model
        self.normalize = normalize
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        features = self.image_model(image)
        return F.normalize(features, dim=-1) if self.normalize else features
    
    def encode_meta(self, meta):
        features = self.meta_model(meta)
        return F.normalize(features, dim=-1) if self.normalize else features
    
    def forward(self, image, meta):
        image_features = self.encode_image(image)
        text_features = self.encode_meta(meta)
        return image_features, text_features, self.logit_scale.exp()