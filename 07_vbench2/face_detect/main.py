
import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")


def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def pad_with_ratio(image, padding_ratio=0.5):
    """
    对图像进行padding，让内容占比变小
    
    Args:
        image: 输入图像
        padding_ratio: padding比例，0.5表示在每个方向添加原尺寸50%的padding
    
    Returns:
        padded_image: padding后的图像
        scale_factor: 缩放因子，用于后续坐标转换
    """
    h, w = image.shape[:2]
    
    # 计算padding大小
    pad_h = int(h * padding_ratio)
    pad_w = int(w * padding_ratio)
    
    # 创建新的图像，填充黑色背景
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    
    # 将原图像放在中央
    padded_image[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    # 计算缩放因子，用于后续坐标转换
    scale_factor = {'pad_w': pad_w, 'pad_h': pad_h, 'orig_w': w, 'orig_h': h}
    
    return padded_image, scale_factor


def adjust_bbox_coordinates(objs, scale_factor, target_w, target_h):
    """
    调整检测框坐标，从padding后的图像坐标转换回原图像坐标
    
    Args:
        objs: 检测结果列表
        scale_factor: 缩放信息
        target_w, target_h: 目标图像尺寸
    
    Returns:
        adjusted_objs: 调整后的检测结果
    """
    if not objs:
        return objs
    
    pad_w = scale_factor['pad_w']
    pad_h = scale_factor['pad_h']
    orig_w = scale_factor['orig_w']
    orig_h = scale_factor['orig_h']
    
    # 计算从padding图像到目标图像的缩放比例
    scale_x = target_w / (orig_w + 2 * pad_w)
    scale_y = target_h / (orig_h + 2 * pad_h)
    
    adjusted_objs = []
    for obj in objs:
        # 调整边界框坐标
        x, y, r, b = obj.box
        
        # 从缩放后的坐标转换为padding图像坐标
        x = x / scale_x
        y = y / scale_y
        r = r / scale_x
        b = b / scale_y
        
        # 从padding图像坐标转换为原图像坐标
        x = x - pad_w
        y = y - pad_h
        r = r - pad_w
        b = b - pad_h
        
        # 确保坐标在原图像范围内
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        r = max(0, min(r, orig_w - 1))
        b = max(0, min(b, orig_h - 1))
        
        # 调整关键点坐标（如果存在）
        adjusted_landmark = None
        if obj.landmark is not None:
            adjusted_landmark = []
            for lm in obj.landmark:
                lm_x, lm_y = lm[:2]
                # 同样的坐标转换过程
                lm_x = lm_x / scale_x - pad_w
                lm_y = lm_y / scale_y - pad_h
                lm_x = max(0, min(lm_x, orig_w - 1))
                lm_y = max(0, min(lm_y, orig_h - 1))
                adjusted_landmark.append((lm_x, lm_y))
        
        # 创建新的BBox对象
        adjusted_obj = common.BBox(
            obj.label, 
            [x, y, r, b], 
            score=obj.score, 
            landmark=adjusted_landmark
        )
        adjusted_objs.append(adjusted_obj)
    
    return adjusted_objs


def detect_single_attempt(model, image, threshold=0.4, nms_iou=0.5):
    """
    单次检测尝试
    """
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    processed_image = common.pad(image)
    processed_image = ((processed_image / 255.0 - mean) / std).astype(np.float32)
    processed_image = processed_image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(processed_image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def detect(model, image, threshold=0.4, nms_iou=0.5, use_padding_retry=True, padding_ratios=[0.3, 0.6, 1.0]):
    """
    改进的检测函数，支持padding重试机制
    
    Args:
        model: DBFace模型
        image: 输入图像
        threshold: 检测阈值
        nms_iou: NMS的IoU阈值
        use_padding_retry: 是否使用padding重试机制
        padding_ratios: padding比例列表，按顺序尝试
    
    Returns:
        objs: 检测结果列表
    """
    # 保存原图像尺寸
    orig_h, orig_w = image.shape[:2]
    
    # 第一次尝试：正常检测
    objs = detect_single_attempt(model, image, threshold, nms_iou)
    
    # 如果检测到人脸或不使用重试机制，直接返回结果
    if objs or not use_padding_retry:
        return objs
    
    print("第一次检测未发现人脸，开始padding重试...")
    
    # 尝试不同的padding比例
    for i, padding_ratio in enumerate(padding_ratios):
        print(f"尝试padding比例: {padding_ratio}")
        
        # 对图像进行padding
        padded_image, scale_factor = pad_with_ratio(image, padding_ratio)
        
        # resize回原图像尺寸
        resized_image = cv2.resize(padded_image, (orig_w, orig_h))
        
        # 进行检测
        retry_objs = detect_single_attempt(model, resized_image, threshold, nms_iou)
        
        if retry_objs:
            print(f"padding重试成功，检测到 {len(retry_objs)} 个人脸")
            # 调整坐标回原图像坐标系
            adjusted_objs = adjust_bbox_coordinates(retry_objs, scale_factor, orig_w, orig_h)
            return adjusted_objs
    
    print("所有padding重试均未检测到人脸")
    return []


def detect_image(model, file, use_padding_retry=True):

    image = common.imread(file)
    objs = detect(model, image, use_padding_retry=use_padding_retry)

    for obj in objs:
        common.drawbbox(image, obj)

    # 添加后缀以区分是否使用了padding重试
    suffix = "_with_padding" if use_padding_retry else "_normal"
    common.imwrite("detect_result/" + common.file_name_no_suffix(file) + suffix + ".draw.jpg", image)
    
    print(f"检测结果保存到: detect_result/{common.file_name_no_suffix(file)}{suffix}.draw.jpg")
    print(f"检测到 {len(objs)} 个人脸")


def image_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    
    # 演示不同的检测模式
    print("=== 演示正常检测模式 ===")
    detect_image(dbface, "datas/selfie.jpg", use_padding_retry=False)
    detect_image(dbface, "datas/12_Group_Group_12_Group_Group_12_728.jpg", use_padding_retry=False)
    
    print("\n=== 演示Padding重试模式 ===")
    detect_image(dbface, "datas/selfie.jpg", use_padding_retry=True)
    detect_image(dbface, "datas/12_Group_Group_12_Group_Group_12_728.jpg", use_padding_retry=True)


def camera_demo(use_padding_retry=True):

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    
    print(f"摄像头演示模式 - Padding重试: {'开启' if use_padding_retry else '关闭'}")
    print("按 'q' 退出，按 't' 切换padding重试模式")

    while ok:
        objs = detect(dbface, frame, use_padding_retry=use_padding_retry)

        for obj in objs:
            common.drawbbox(frame, obj)

        # 在图像上显示检测信息
        info_text = f"Faces: {len(objs)}, Padding Retry: {'ON' if use_padding_retry else 'OFF'}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 't' to toggle padding retry", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_padding_retry = not use_padding_retry
            print(f"Padding重试模式切换为: {'开启' if use_padding_retry else '关闭'}")

        ok, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()

def test_padding_feature():
    """
    测试padding重试功能的效果
    """
    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    
    # 可以在这里放置一个测试图像路径
    test_image_path = "datas/selfie.jpg"  # 修改为实际的测试图像路径
    
    if os.path.exists(test_image_path):
        print("=== 测试Padding重试功能 ===")
        image = common.imread(test_image_path)
        
        # 正常检测
        print("\n1. 正常检测:")
        objs_normal = detect(dbface, image, use_padding_retry=False)
        print(f"   检测到 {len(objs_normal)} 个人脸")
        
        # 使用padding重试
        print("\n2. 使用Padding重试:")
        objs_with_padding = detect(dbface, image, use_padding_retry=True)
        print(f"   检测到 {len(objs_with_padding)} 个人脸")
        
        if len(objs_with_padding) > len(objs_normal):
            print("✓ Padding重试功能有效！检测到了更多人脸。")
        elif len(objs_with_padding) == len(objs_normal) and len(objs_normal) > 0:
            print("= 两种方法检测结果相同。")
        else:
            print("- 在这个图像上，Padding重试没有带来额外的检测结果。")
    else:
        print(f"测试图像 {test_image_path} 不存在，跳过测试。")


if __name__ == "__main__":
    import os
    
    # 你可以选择运行哪个演示
    print("请选择要运行的演示:")
    print("1. 图像演示 (image_demo)")
    print("2. 摄像头演示 (camera_demo)")
    print("3. 测试Padding功能 (test_padding_feature)")
    print("4. 运行所有演示")
    
    choice = input("请输入选择 (1-4, 默认为4): ").strip()
    
    if choice == "1":
        image_demo()
    elif choice == "2":
        camera_demo()
    elif choice == "3":
        test_padding_feature()
    else:
        # 默认运行所有演示
        image_demo()
        camera_demo()
        test_padding_feature()
    


    