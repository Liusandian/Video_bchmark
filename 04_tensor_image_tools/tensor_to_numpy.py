import torch
import numpy as np

# Tensor to NumPy
tensor = torch.randn(3, 224, 224)
numpy_array = tensor.numpy()                    # CPU tensor
numpy_array = tensor.detach().cpu().numpy()     # GPU tensor (推荐)

# NumPy to Tensor
numpy_array = np.random.randn(3, 224, 224)
tensor = torch.from_numpy(numpy_array)          # 共享内存
tensor = torch.tensor(numpy_array)              # 复制数据