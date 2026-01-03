import torch
import torch.nn as nn

class SimpleSplitCNN(nn.Module):
    def __init__(self):
        super(SimpleSplitCNN, self).__init__()
        # Define layers individually to allow splitting
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Layer 0: ~high potential data reduction if kept, but compute heavy? No, early layers often expand data.
            nn.ReLU(),                                  # Layer 1
            nn.MaxPool2d(2),                            # Layer 2: Reduces data size 4x
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# Layer 3
            nn.ReLU(),                                  # Layer 4
            nn.MaxPool2d(2),                            # Layer 5: Reduces data size 4x
            nn.Flatten(),                               # Layer 6
            nn.Linear(64 * 8 * 8, 128),                 # Layer 7: (Assuming 32x32 input)
            nn.ReLU(),                                  # Layer 8
            nn.Linear(128, 10)                          # Layer 9
        ])
        
        # Profiled FLOPS per layer (Mock values for simulation)
        # In reality these would be measured.
        # Format: (Layer Index: GFLOPs)
        self.layer_complexity = {
            0: 0.1, 1: 0.01, 2: 0.01,
            3: 0.4, 4: 0.04, 5: 0.01,
            6: 0.0, 7: 0.2, 8: 0.01, 9: 0.01
        }
        
        # Output Data Size Scale Factor relative to Input
        # Approximate values
        self.data_volume = {
            -1: 1.0, # Input
            0: 32.0, # 3->32 channels
            1: 32.0,
            2: 8.0,  # MaxPool 2x2 reduces spatial dim by 4, volume by 4. So 32/4 = 8.
            3: 16.0, # 32->64 channels. 8*2 = 16.
            4: 16.0,
            5: 4.0,  # MaxPool 2x2. 16/4 = 4.
            6: 4.0,
            7: 0.1,  # Dense layers have small outputs usually compared to feature maps
            8: 0.1,
            9: 0.01
        }

    def forward(self, x, start_layer=0, end_layer=None):
        if end_layer is None:
            end_layer = len(self.layers)
            
        for i in range(start_layer, end_layer):
            x = self.layers[i](x)
            
        return x

    def get_layer_stats(self, layer_idx):
        """Returns (flops, output_data_ratio)"""
        return self.layer_complexity.get(layer_idx, 0), self.data_volume.get(layer_idx, 1)
