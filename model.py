import torch.nn as nn
import torchvision.models as models

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        
        # Use a pre-trained ResNet-50 model
        self.base_model = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer to get features
        # This will give us the feature of size 2048 (from the last avgpool layer)
        modules = list(self.base_model.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        
        # Now add our regression head with an additional ReLU after the final linear layer
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),  # Three outputs: gamma_bright, gamma_dark and saturation
            nn.ReLU()           # Additional ReLU to ensure non-negative outputs
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.head(x)
        return x

model = RegressionModel()
