
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize
from torchvision.transforms import functional as F

def visualize_image(image_tensor):
    """
    Visualize a PyTorch image tensor
    
    Args:
    - image_tensor: A torch tensor of shape [3, 224, 224] or [1, 3, 224, 224]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    image_np = image_tensor.cpu().numpy()

    image_np = np.transpose(image_np, (1, 2, 0))
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_np = image_np * std + mean
    
    # Clip values to 0-1 range
    image_np = np.clip(image_np, 0, 1)
    
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def letterbox_to_square(img, size=256, fill=0):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale * 0.8)), int(round(h * scale * 0.8))
    img = F.resize(img, (new_h, new_w), antialias=True)
    
    pad_left   = (size - new_w) // 2
    pad_right  = size - new_w - pad_left
    pad_top    = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)
    return img
