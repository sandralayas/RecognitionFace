import torch
from torch import nn

class AgeEstimator(nn.Module):
    def __init__(self, backbone):
        super(AgeEstimator, self).__init__()
        
        # The pre-trained IResNet-100 backbone
        self.backbone = backbone
        
        # Freeze the backbone layers so their weights are not updated during training
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # The new output layers (the "head")
        # The backbone outputs a 512-dimensional feature vector.
        self.head = nn.Sequential(
            nn.Linear(512, 256),       # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),           # Dropout for regularization
            nn.Linear(256, 128),      # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)          # Final output layer for a single age value (regression)
        )

    def forward(self, x):
        # Pass the input through the frozen backbone
        with torch.no_grad():
            features = self.backbone(x)
            
        # Pass the extracted features through the new head
        age_prediction = self.head(features)
        
        # Reshape the output to be a 1-dimensional tensor
        return age_prediction.squeeze(1)