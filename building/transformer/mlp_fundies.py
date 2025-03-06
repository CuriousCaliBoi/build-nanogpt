import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

class LinearProjection(nn.Module):
    """A simple linear projection layer in PyTorch"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

def analyze_linear_projection(batch_sizes, in_features, out_features):
    """Analyze computational cost and memory usage of a linear projection"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearProjection(in_features, out_features).to(device)
    
    # Calculate theoretical parameters and FLOPs
    param_count = in_features * out_features + out_features  # weights + bias
    
    results = {
        'batch_size': [],
        'inference_time': [],
        'memory_usage': [],
        'theoretical_flops': []
    }
    
    for batch_size in batch_sizes:
        # Create input tensor
        x = torch.randn(batch_size, in_features).to(device)
        
        # Measure inference time
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        # Run multiple iterations for more accurate timing
        iterations = 100
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Calculate memory usage
        memory_bytes = (
            # Parameters
            param_count * 4 +  
            # Input tensor
            batch_size * in_features * 4 +  
            # Output tensor
            batch_size * out_features * 4
        )
        memory_mb = memory_bytes / (1024 * 1024)
        
        # Calculate FLOPs (multiply-adds)
        # Each output element requires in_features multiplications and in_features-1 additions
        flops_per_output = 2 * in_features - 1 + 1  # multiplications + additions + bias
        total_flops = batch_size * out_features * flops_per_output
        
        # Store results
        results['batch_size'].append(batch_size)
        results['inference_time'].append(avg_time * 1000)  # convert to ms
        results['memory_usage'].append(memory_mb)
        results['theoretical_flops'].append(total_flops / 1e6)  # convert to MFLOPs
    
    return results, param_count

def plot_results(results):
    """Plot the analysis results"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot inference time
    axs[0].plot(results['batch_size'], results['inference_time'], 'o-', color='blue')
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Inference Time (ms)')
    axs[0].set_title('Inference Time vs Batch Size')
    axs[0].grid(True)
    
    # Plot memory usage and FLOPs
    ax1 = axs[1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(results['batch_size'], results['memory_usage'], 'o-', color='green', label='Memory Usage')
    line2 = ax2.plot(results['batch_size'], results['theoretical_flops'], 'o-', color='red', label='Theoretical FLOPs')
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Memory Usage (MB)', color='green')
    ax2.set_ylabel('Theoretical FLOPs (M)', color='red')
    ax1.set_title('Memory Usage and Computational Cost vs Batch Size')
    ax1.grid(True)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('linear_projection_analysis.png')
    plt.show()

if __name__ == "__main__":
    # Define parameters for analysis
    in_features = 768    # Typical embedding dimension in transformers
    out_features = 3072  # 4x expansion common in MLP blocks

