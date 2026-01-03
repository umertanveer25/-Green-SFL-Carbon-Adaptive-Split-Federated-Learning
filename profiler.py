import torch
import time
import numpy as np
import pandas as pd
from iot_model import RealIoTCNN

class ModelProfiler:
    def __init__(self, model_class=RealIoTCNN):
        self.model = model_class()
        self.model.eval()
        
        # We profile the logical "Blocks" defined in the model's self.layers
        self.layers = list(self.model.layers)

    def profile(self, input_shape=(1, 47)):
        """
        Runs a dummy input through the model logical blocks.
        Measures: Output Size (MB), Latency (ms).
        """
        x = torch.randn(input_shape)
        results = []
        
        print(f"Profiling RealIoTCNN with input {input_shape}...")
        
        # Initial input formatting (Unsqueeze if needed matching model logic)
        if x.dim() == 2:
            x = x.unsqueeze(1) # [Batch, 1, 47]
            
        for i, layer in enumerate(self.layers):
            # Measure Latency
            # Warmup
            _ = layer(x)
            
            # Run
            iters = 50 # More iterations for small layers
            t0 = time.time()
            for _ in range(iters):
                out = layer(x)
            t1 = time.time()
            
            avg_latency_ms = ((t1 - t0) / iters) * 1000
            
            # Measure Output Size (Activations to be transmitted)
            out_shape = out.shape
            # Float32 = 4 bytes
            size_bytes = np.prod(out_shape) * 4 
            size_mb = size_bytes / (1024 * 1024)
            
            results.append({
                "layer_idx": i,
                "layer_name": f"block_{i}",
                "output_shape": str(list(out_shape)),
                "output_mb": size_mb,
                "compute_ms": avg_latency_ms
            })
            
            x = out # Pass to next layer
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    profiler = ModelProfiler()
    df = profiler.profile()
    df.to_csv("iot_model_profile.csv", index=False)
    print(df)
