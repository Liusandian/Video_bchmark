def print_tensor_info(tensor, name="Tensor"):
    """打印张量信息"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print()

def visualize_tensor(tensor, title="Tensor"):
    """可视化张量"""
    import matplotlib.pyplot as plt
    
    # 转换为可显示格式
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一个batch
    
    if tensor.dim() == 3:
        if tensor.shape[0] == 3:  # CHW格式
            tensor = tensor.permute(1, 2, 0)
    
    # 确保在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 转换为numpy
    np_image = tensor.detach().numpy()
    
    # 调整数值范围
    if np_image.max() <= 1.0:
        np_image = np_image
    else:
        np_image = np_image / 255.0
    
    plt.figure(figsize=(8, 6))
    plt.imshow(np_image)
    plt.title(title)
    plt.axis('off')
    plt.show()