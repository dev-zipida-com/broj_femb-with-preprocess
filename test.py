import os
import torch


checkpoint_path = 'checkpoint/checkpoint.pt'
if os.path.exists(checkpoint_path):
    loaded_checkpoint = torch.load(checkpoint_path)
    print(loaded_checkpoint.keys())
    print(loaded_checkpoint['epoch'])
    print(loaded_checkpoint['trained_epoch'])