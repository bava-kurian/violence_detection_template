import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

def get_model():
    model = r3d_18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model

