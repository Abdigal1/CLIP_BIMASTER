import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from torch import nn

import argostranslate.translate
import nlpaug.augmenter.word as naw
from httpcore._exceptions import ReadTimeout
import numpy as np

class Bertpt(nn.Module):
    
    def __init__(self, output_dim: int):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
        checkpoint_path = "/share_zeta/DeepOil/deepoil/Code/CLIP/notebooks/old-pretrained-bert/checkpoint-26000/"
        self.bert = BertForSequenceClassification.from_pretrained(checkpoint_path,
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
        
        self.freeze(top_layers=20)
    
    def freeze(self, top_layers=4):
        total_layers = len(list(self.bert.parameters()))
        for i, param in enumerate(self.bert.parameters()):
            if i < total_layers - top_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
               
    def unfreeze(self):
        total_layers = len(list(self.bert.parameters()))
        for i, param in enumerate(self.bert.parameters()):
                param.requires_grad = True



class SynAug(object):
    """Replace words by its synonym in english and then backtranslates to portuguese
        input: list of strings
        output: list of strings augmented

    Args:
        ratio: Percentage of text to be replaced
        p: Percentage of the augmentation to occur
    """

    def __init__(self, p=0.5, ratio=0.3):
        self.src = 'pt'
        self.dest = 'en'
        self.p = p
        self.aug = naw.SynonymAug(aug_src='wordnet', aug_p=ratio)

    def __call__(self, sample_):
        if np.random.uniform()<self.p:
            sample_eng = argostranslate.translate.translate(sample_, self.src, self.dest)
            sample_sim = self.aug.augment(sample_eng)
            sample_sim = sample_sim[0]
            sample_pt = argostranslate.translate.translate(sample_sim, self.dest, self.src)
            return sample_pt
        else:
            return sample_