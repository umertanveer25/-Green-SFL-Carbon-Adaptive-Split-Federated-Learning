import torch
import torch.nn as nn

class RealIoTCNN(nn.Module):
    def __init__(self, input_features=47, num_classes=8): # 7 attacks + 1 benign
        super(RealIoTCNN, self).__init__()
        # CICIoT 2023 is tabular/1D flow data. 
        # We use a 1D CNN + MLP architecture common in Network Intrusion Detection (NIDS).
        
        # Layer 0: Input -> Embed/Conv
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Layer 1: Learn features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Layer 2: Flatten & Dense
        self.flatten = nn.Flatten()
        
        # Calculate flat size: Input 47 -> Pool(2) -> ~23 -> Pool(2) -> ~11
        # 64 channels * 11 = 704
        self.fc1 = nn.Linear(64 * 11, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
        # Logical Blocks for Split Learning
        self.layers = nn.ModuleList([
            nn.Sequential(self.conv1, self.relu1), # Block 0
            self.pool1,                            # Block 1
            nn.Sequential(self.conv2, self.relu2), # Block 2
            self.pool2,                            # Block 3
            self.flatten,                          # Block 4
            nn.Sequential(self.fc1, self.relu3),   # Block 5
            self.fc2                               # Block 6
        ])
        
        # Mock Profile Stats (To be replaced by real profiler)
        self.layer_complexity = {}
        self.data_volume = {}

    def forward(self, x):
        # x shape: [Batch, Features] -> [Batch, 1, Features] for Conv1D
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        for layer in self.layers:
            x = layer(x)
        return x
