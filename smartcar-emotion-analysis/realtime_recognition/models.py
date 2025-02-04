import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# MobileNetV3Net 정의
class MobileNetV3Net(nn.Module):
    def __init__(self, num_emotions=2, dropout_rate=0.2):
        super(MobileNetV3Net, self).__init__()
        # self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_emotions)
        )

    def forward(self, x):
        features = self.backbone(x)
        x = self.fc(features)
        return x

def model1(model1_path, device):
    model1 = MobileNetV3Net(num_emotions=2)
    model1.load_state_dict(torch.load(model1_path, map_location=device, weights_only=True))
    model1.to(device)
    model1.eval()

    return model1


def model2(model2_path, device):
    model2 = MobileNetV3Net(num_emotions=2)
    model2.load_state_dict(torch.load(model2_path, map_location=device, weights_only=True))
    model2.to(device)
    model2.eval()

    return model2