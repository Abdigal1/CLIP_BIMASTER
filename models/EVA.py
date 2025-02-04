import torch
from torch import nn
import timm

class EVA_base(nn.Module):
    
    def __init__(self, output_dim, feature_extracting:bool=True, pretrained:bool=True):
        super().__init__()
        self.model = timm.create_model('eva02_base_patch14_224',pretrained=pretrained,num_classes=0,in_chans=6)
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # in_features = self.model.head.in_features
        self.model.head = nn.Linear(768, output_dim)

        #EXTRA
        self.freeze()

    def forward(self, x):
        
        logits = self.model(x)
        
        return logits

    def freeze(self, top_layers=23):
        total_layers = len(list(self.model.parameters()))
        for i, param in enumerate(self.model.parameters()):
            if i < total_layers - top_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
class EVA_large(nn.Module):
    
    def __init__(self, output_dim, feature_extracting:bool=True, pretrained:bool=True):
        super().__init__()
        self.model = timm.create_model('eva02_large_patch14_224',pretrained=pretrained,num_classes=0,in_chans=6)
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # in_features = self.model.head.in_features
        self.model.head = nn.Linear(1024, output_dim)

    def forward(self, x):
        
        logits = self.model(x)
        
        return logits