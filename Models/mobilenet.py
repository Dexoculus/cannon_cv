# models/mobilenet_module.py

import torch
import torch.nn as nn
import torchvision.models as models

class CustomMobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initializes a CustomMobileNetV3Small model.

        Args:
            num_classes (int): The number of output classes for the new classifier.
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
                               Uses new 'weights' API if available (torchvision >= 0.13),
                               otherwise falls back to 'pretrained=True'.
        """
        super(CustomMobileNetV3Small, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        weights_arg = None
        model_kwargs = {}

        if self.pretrained:
            try:
                weights_arg = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                model_kwargs['weights'] = weights_arg
                print("Using MobileNet_V3_Small_Weights.IMAGENET1K_V1 for pretrained weights.")
            except AttributeError:
                model_kwargs['pretrained'] = True
                print("Using legacy 'pretrained=True' for MobileNetV3-Small (torchvision < 0.13 or weights enum not found).")
        else:
            model_kwargs['pretrained'] = False # Or weights=None for newer torchvision if default weights are not desired

        # Load the base MobileNetV3-Small model
        self.base_model = models.mobilenet_v3_small(**model_kwargs)

        # Modify the classifier
        # The classifier in MobileNetV3 is a nn.Sequential module.
        # The last layer of this nn.Sequential is the nn.Linear layer we want to replace.
        num_ftrs = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.base_model(x)

    def get_input_transforms(self):
        """
        Returns the recommended input transformations for the pretrained model if available.
        Returns None if not using new weights API or not pretrained.
        """
        if self.pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                return weights.transforms()
            except AttributeError:
                print("Input transforms not automatically available via old 'pretrained=True' API. "
                      "Use standard ImageNet transforms (e.g., resize to 224, normalize).")
                return None # Or return a default transform
        return None