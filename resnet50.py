import torch.nn as nn
from torchvision import models

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class ResNet50Model(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # if pretrained:
        #     for param in self.resnet50.parameters():
        #         param.requires_grad = False
        
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet50(x)