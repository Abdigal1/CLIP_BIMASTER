import torch
from torch import nn
import timm

class ViT(nn.Module):
    
    def __init__(self, output_dim, feature_extracting:bool=True, pretrained:str='imagenet'):
        super().__init__()
        self.model = None
        if pretrained=='imagenet' or pretrained=='custom':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=6)
        else:
            self.model = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=6)

        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4),
        )

        self.model
        if pretrained=='custom':
            checkpoint = torch.load(r'/share_zeta/DeepOil/repo_geilson/models/model_0.71.pt')
            for key in list(checkpoint.keys()):
                if 'model.' in key:
                    checkpoint[key.replace('model.', '')] = checkpoint[key]
                    del checkpoint[key]
            self.model.load_state_dict(checkpoint)

        self.model.head = nn.Linear(in_features, output_dim)
        self.freeze()

    def forward(self, x):
        
        logits = self.model(x)
        
        return logits
    
    def freeze(self, top_layers=16):
        total_layers = len(list(self.model.parameters()))
        for i, param in enumerate(self.model.parameters()):
            if i < total_layers - top_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True