import torch
from z_modelops import NameToLanguages, load_labels
from z_dataops import transform
import json
from torch import nn

def load_model(location="model/rnn.pth"): 
    '''loads the model, together with arch'''
    model = torch.load(location, weights_only=False) 
    return model

def infer_lang(name:str, model, label:dict, k=3)-> str:
    name_tensor = transform(name)
    with torch.no_grad():
        logits = model(name_tensor.unsqueeze(0))
        y_pred = nn.Softmax(dim=1)(logits)
    top_k_idx = y_pred.sort(descending=True, dim=1).indices.numpy()[0][:k]
    return [label[str(idx)] for idx in top_k_idx]

def setup_inference():
    # load model 
    model = load_model()
    # call the model with inputs 
    labels = load_labels()
    return model, labels


if __name__=="__main__":
    model, labels = setup_inference()
    