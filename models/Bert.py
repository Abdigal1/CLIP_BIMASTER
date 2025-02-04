import torch
from torch import nn

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class Bert(nn.Module):
    
    def __init__(self, output_dim: int):
        super().__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                   num_labels = output_dim)
        

    def forward(self, x):
        tokens = self.tokenizer.batch_encode_plus(x,
                          add_special_tokens=True,
                          return_tensors="pt", 
                          truncation=True, 
                          max_length=512,
                          padding=True)
        tokens['input_ids'] = tokens['input_ids'].cuda(non_blocking=True)
        tokens['attention_mask'] = tokens['attention_mask'].cuda(non_blocking=True)
        output = self.bert(input_ids=tokens['input_ids'],
                        attention_mask=tokens['attention_mask'])
        
        return output['logits']
    
    def freeze(self, top_layers=3):
        total_layers = len(list(self.bert.parameters()))
        for i, param in enumerate(self.bert.parameters()):
            if i < total_layers - top_layers:
                param.requires_grad = False
               
    def unfreeze(self):
        total_layers = len(list(self.bert.parameters()))
        for i, param in enumerate(self.bert.parameters()):
                param.requires_grad = True