import random
import numpy as np
import torch
from typing import Dict

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✓ Seed set to {seed}")

def get_device_info() -> Dict[str, any]:
    """Get GPU device information"""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    
    device_id = 0
    device = torch.cuda.get_device_properties(device_id)
    
    return {
        "device": "cuda",
        "name": torch.cuda.get_device_name(device_id),
        "total_memory_gb": device.total_memory / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated(device_id) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device_id) / 1024**3
    }

def print_device_info():
    """Print GPU information"""
    info = get_device_info()
    
    print("\n" + "="*80)
    print("DEVICE INFORMATION")
    print("="*80)
    
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("="*80 + "\n")