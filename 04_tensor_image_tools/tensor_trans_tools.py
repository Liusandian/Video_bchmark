# 维度调整
from image_process import ImageConverter
import torch
import numpy as np
import cv2
from torchvision import transforms
from einops import rearrange
from PIL import Image

# 1. 添加/移除维度
tensor = torch.randn(224, 224, 3)
tensor = tensor.unsqueeze(0)        # 添加batch维度: (1, 224, 224, 3)
tensor = tensor.squeeze(0)          # 移除batch维度: (224, 224, 3)

# 2. 重塑维度
tensor = tensor.view(-1, 3, 224, 224)    # reshape
tensor = tensor.reshape(-1, 3, 224, 224) # 同上，更安全

# 3. 连续内存
tensor = tensor.contiguous()        # 确保内存连续

# 4. 克隆张量
tensor_copy = tensor.clone()        # 深拷贝
tensor_detach = tensor.detach()     # 分离计算图


# 批量处理张量

def batch_process_images(image_list, device='cuda'):
    """批量处理图像"""
    tensors = []
    for img in image_list:
        if isinstance(img, Image.Image):
            tensor = ImageConverter.pil_to_tensor(img)
        elif isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    
    # 堆叠成批次
    batch_tensor = torch.stack(tensors).to(device)
    return batch_tensor

def unbatch_tensors(batch_tensor):
    """拆分批次张量"""
    return [tensor for tensor in batch_tensor]