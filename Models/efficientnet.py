# models/efficientnet_module.py

import torch
import torch.nn as nn
import timm

class CustomEfficientNetLite0(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, model_name: str = 'tf_efficientnet_lite0'):
        """
        Initializes a CustomEfficientNetLite0 model.

        Args:
            num_classes (int): The number of output classes for the new classifier.
            pretrained (bool): If True, loads weights pre-trained on ImageNet from timm.
            model_name (str): The name of the model in the timm library
                              (e.g., 'tf_efficientnet_lite0', 'efficientnet_lite0').
        """
        super(CustomEfficientNetLite0, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_name = model_name

        try:
            # Load the base EfficientNet-Lite0 model using timm
            self.base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
            print(f"Loaded '{self.model_name}' with pretrained={self.pretrained} using timm.")
        except Exception as e:
            print(f"Error creating model '{self.model_name}' with timm: {e}")
            print("Please ensure 'timm' library is installed (`pip install timm`) and the model name is correct.")
            raise

        # Modify the classifier
        # The classifier in timm's EfficientNet models is typically named 'classifier'.
        # For some other timm models it might be 'fc'.
        if hasattr(self.base_model, 'classifier'):
            num_ftrs = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(num_ftrs, self.num_classes)
        elif hasattr(self.base_model, 'fc'): # Fallback for models using 'fc' as classifier name
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, self.num_classes)
        else:
            # print(self.base_model) # Uncomment to inspect model structure
            raise AttributeError(f"Could not find a standard classifier attribute ('classifier' or 'fc') on model '{self.model_name}'. "
                                 "Please inspect the model structure and adapt the modification.")
    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.base_model(x)

    def get_input_cfg(self):
        """
        Returns the default_cfg of the timm model, which contains input_size, mean, std etc.
        """
        if hasattr(self.base_model, 'default_cfg'):
            return self.base_model.default_cfg
        return None