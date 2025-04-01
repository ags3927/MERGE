import torch
import numpy as np

def recursively_serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [recursively_serialize(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: recursively_serialize(v) for k, v in obj.items()}
    else:
        return obj