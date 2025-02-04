import torch
from torch import nn
import torch.nn.functional as F

class ClipLoss(nn.Module):

    def __init__(self, label_smoothing):
        super().__init__()
        self.label_smoothing = label_smoothing


    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text
    
    
    def forward(self, image_features, text_features, logit_scale, p_labels=None):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        total_loss = None
        if p_labels is not None:
            labels = p_labels
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        else:
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            
            total_loss = (
                F.cross_entropy(logits_per_image, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits_per_text, labels, label_smoothing=self.label_smoothing)
            ) / 2
        if torch.isnan(total_loss):
            print(torch.isnan(image_features).any(), torch.isnan(text_features).any())
        
        return  total_loss

class ClipLoss_v2(nn.Module):

    def __init__(self):
        super().__init__()


    def get_ground_truth(self, device, num_logits, descriptions) -> torch.Tensor:
        labels = torch.zeros(num_logits, dtype=torch.long)
        labels_dict = {}
        count=0
        for i in range(num_logits):
            if not(descriptions[i] in labels_dict.keys()):
                labels_dict[descriptions[i]] = count
                labels[i] = count
                count+=1
            else:
                labels[i] = labels_dict[descriptions[i]]
        labels = labels.to(device)
        # print(labels)
        return labels

        
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text
    
    
    def forward(self, image_features, text_features, logit_scale, descriptions):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        total_loss = None

        labels = self.get_ground_truth(device, logits_per_image.shape[0], descriptions)
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        if torch.isnan(total_loss):
            print(torch.isnan(image_features).any(), torch.isnan(text_features).any())
        
        return  total_loss