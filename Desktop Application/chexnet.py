import numpy as np
from torchvision.models import densenet121
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch.autograd.variable import Variable
from layers import Flatten

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class ChexNet(nn.Module):
    tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])

    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        self.backbone = densenet121(False).features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 Flatten(),
                                 nn.Linear(1024, 14))
        model_name = 'model.h5'
        state_dict = torch.load(model_name)
        self.load_state_dict(state_dict)


    def forward(self, x):
        return self.head(self.backbone(x))

    def predict(self, image):
        """
        input: PIL image (w, h, c)
        output: prob np.array
        """
        image = Variable(self.tfm(image)[None])
        py = torch.sigmoid(self(image))
        prob = py.detach().cpu().numpy()[0]
        return prob
