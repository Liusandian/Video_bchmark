import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from einops import rearrange

class ImageConverter:
    """图像格式转换工具类"""
    
    @staticmethod
    def pil_to_tensor(pil_image, normalize=True):
        """PIL Image -> PyTorch Tensor"""
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),  # 自动 /255 和 HWC->CHW
            ])
            return transform(pil_image)
        else:
            # 手动转换
            np_image = np.array(pil_image)
            if len(np_image.shape) == 3:
                tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(np_image).unsqueeze(0)
            return tensor.float()
    
    @staticmethod
    def tensor_to_pil(tensor):
        """PyTorch Tensor -> PIL Image"""
        # 确保在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 处理批次维度
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除batch维度
        
        # CHW -> HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 转换为numpy并确保数据类型
        np_image = tensor.detach().numpy()
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)
    
    @staticmethod
    def cv2_to_tensor(cv2_image, normalize=True):
        """OpenCV Image -> PyTorch Tensor"""
        # BGR -> RGB
        if len(cv2_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2_image
        
        # HWC -> CHW
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        
        if normalize:
            tensor = tensor / 255.0
        
        return tensor
    
    @staticmethod
    def tensor_to_cv2(tensor):
        """PyTorch Tensor -> OpenCV Image"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 处理批次维度
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # CHW -> HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 转换为numpy
        np_image = tensor.detach().numpy()
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        
        # RGB -> BGR
        if len(np_image.shape) == 3:
            bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = np_image
        
        return bgr_image