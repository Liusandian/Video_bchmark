
import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")


def calculate_face_ratio(objs, image_width, image_height):
    """
    计算人脸占图像的比例
    
    Args:
        objs: 检测到的人脸列表
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        face_ratio_info: 包含人脸占比信息的字典
    """
    if not objs:
        return {
            'total_face_area': 0,
            'image_area': image_width * image_height,
            'face_ratio': 0.0,
            'face_count': 0,
            'largest_face_ratio': 0.0,
            'average_face_ratio': 0.0
        }
    
    image_area = image_width * image_height
    total_face_area = 0
    face_areas = []
    
    for obj in objs:
        face_area = obj.area
        total_face_area += face_area
        face_areas.append(face_area)
    
    face_ratio = total_face_area / image_area
    largest_face_ratio = max(face_areas) / image_area if face_areas else 0.0
    average_face_ratio = (total_face_area / len(objs)) / image_area if objs else 0.0
    
    return {
        'total_face_area': total_face_area,
        'image_area': image_area,
        'face_ratio': face_ratio,
        'face_count': len(objs),
        'largest_face_ratio': largest_face_ratio,
        'average_face_ratio': average_face_ratio,
        'face_areas': face_areas
    }


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


def detect(model, image, threshold=0.4, nms_iou=0.5, use_padding_retry=True, padding_ratios=[0.3, 0.6, 1.0], return_face_ratio=True):
    """
    改进的检测函数，支持padding重试机制和人脸占比计算
    
    Args:
        model: DBFace模型
        image: 输入图像
        threshold: 检测阈值
        nms_iou: NMS的IoU阈值
        use_padding_retry: 是否使用padding重试机制
        padding_ratios: padding比例列表，按顺序尝试
        return_face_ratio: 是否返回人脸占比信息
    
    Returns:
        如果return_face_ratio=True:
            (objs, detection_info): 检测结果列表和检测信息字典
        如果return_face_ratio=False:
            objs: 检测结果列表（保持向后兼容）
    """
    # 保存原图像尺寸
    orig_h, orig_w = image.shape[:2]
    
    # 初始化检测信息
    detection_info = {
        'used_padding': False,
        'padding_ratio_used': None,
        'original_face_ratio': None,
        'detection_attempts': 0,
        'original_image_size': (orig_w, orig_h)
    }
    
    # 第一次尝试：正常检测
    detection_info['detection_attempts'] += 1
    objs = detect_single_attempt(model, image, threshold, nms_iou)
    
    # 计算原图人脸占比
    if objs:
        face_ratio_info = calculate_face_ratio(objs, orig_w, orig_h)
        detection_info['original_face_ratio'] = face_ratio_info
        detection_info['face_ratio_info'] = face_ratio_info
        
        if return_face_ratio:
            return objs, detection_info
        else:
            return objs
    
    # 如果没有检测到人脸且不使用重试机制，返回空结果
    if not use_padding_retry:
        detection_info['original_face_ratio'] = calculate_face_ratio([], orig_w, orig_h)
        detection_info['face_ratio_info'] = detection_info['original_face_ratio']
        
        if return_face_ratio:
            return [], detection_info
        else:
            return []
    
    print("第一次检测未发现人脸，开始padding重试...")
    detection_info['used_padding'] = True
    
    # 尝试不同的padding比例
    for i, padding_ratio in enumerate(padding_ratios):
        print(f"尝试padding比例: {padding_ratio}")
        detection_info['detection_attempts'] += 1
        
        # 对图像进行padding
        padded_image, scale_factor = pad_with_ratio(image, padding_ratio)
        
        # resize回原图像尺寸
        resized_image = cv2.resize(padded_image, (orig_w, orig_h))
        
        # 进行检测
        retry_objs = detect_single_attempt(model, resized_image, threshold, nms_iou)
        
        if retry_objs:
            print(f"padding重试成功，检测到 {len(retry_objs)} 个人脸")
            detection_info['padding_ratio_used'] = padding_ratio
            
            # 调整坐标回原图像坐标系
            adjusted_objs = adjust_bbox_coordinates(retry_objs, scale_factor, orig_w, orig_h)
            
            # 计算调整后的人脸占比（基于原图像尺寸）
            face_ratio_info = calculate_face_ratio(adjusted_objs, orig_w, orig_h)
            detection_info['original_face_ratio'] = face_ratio_info
            detection_info['face_ratio_info'] = face_ratio_info
            
            if return_face_ratio:
                return adjusted_objs, detection_info
            else:
                return adjusted_objs
    
    print("所有padding重试均未检测到人脸")
    detection_info['original_face_ratio'] = calculate_face_ratio([], orig_w, orig_h)
    detection_info['face_ratio_info'] = detection_info['original_face_ratio']
    
    if return_face_ratio:
        return [], detection_info
    else:
        return []


def detect_simple(model, image, threshold=0.4, nms_iou=0.5, use_padding_retry=True, padding_ratios=[0.3, 0.6, 1.0]):
    """
    简化的检测函数，仅返回检测结果（向后兼容）
    
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
    return detect(model, image, threshold, nms_iou, use_padding_retry, padding_ratios, return_face_ratio=False)


def detect_image(model, file, use_padding_retry=True):

    image = common.imread(file)
    objs, detection_info = detect(model, image, use_padding_retry=use_padding_retry, return_face_ratio=True)

    for obj in objs:
        common.drawbbox(image, obj)

    # 添加后缀以区分是否使用了padding重试
    suffix = "_with_padding" if use_padding_retry else "_normal"
    common.imwrite("detect_result/" + common.file_name_no_suffix(file) + suffix + ".draw.jpg", image)
    
    print(f"检测结果保存到: detect_result/{common.file_name_no_suffix(file)}{suffix}.draw.jpg")
    print(f"检测到 {len(objs)} 个人脸")
    if detection_info['face_ratio_info']:
        print(f"人脸占比信息: {detection_info['face_ratio_info']}")


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
        objs, detection_info = detect(dbface, frame, use_padding_retry=use_padding_retry, return_face_ratio=True)

        for obj in objs:
            common.drawbbox(frame, obj)

        # 在图像上显示检测信息
        info_text = f"Faces: {len(objs)}, Padding Retry: {'ON' if use_padding_retry else 'OFF'}"
        if detection_info['face_ratio_info']:
            info_text += f", Face Ratio: {detection_info['face_ratio_info']['face_ratio']:.1%}"
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

def print_face_ratio_details(detection_info, method_name=""):
    """
    详细打印人脸占比信息
    
    Args:
        detection_info: 检测信息字典
        method_name: 检测方法名称
    """
    if not detection_info['face_ratio_info']:
        print(f"   {method_name} - 未检测到人脸")
        return
    
    ratio_info = detection_info['face_ratio_info']
    print(f"   {method_name} - 详细信息:")
    print(f"     检测到人脸数量: {ratio_info['face_count']}")
    print(f"     图像尺寸: {detection_info['original_image_size'][0]}x{detection_info['original_image_size'][1]}")
    print(f"     图像总面积: {ratio_info['image_area']:,} 像素")
    print(f"     人脸总面积: {ratio_info['total_face_area']:,.0f} 像素")
    print(f"     人脸总占比: {ratio_info['face_ratio']:.2%}")
    print(f"     最大人脸占比: {ratio_info['largest_face_ratio']:.2%}")
    print(f"     平均人脸占比: {ratio_info['average_face_ratio']:.2%}")
    
    if detection_info['used_padding']:
        print(f"     使用了Padding重试: 是 (比例: {detection_info['padding_ratio_used']})")
    else:
        print(f"     使用了Padding重试: 否")
    
    print(f"     检测尝试次数: {detection_info['detection_attempts']}")


def analyze_face_size_distribution(objs, image_width, image_height):
    """
    分析人脸尺寸分布
    
    Args:
        objs: 检测到的人脸列表
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        analysis: 人脸尺寸分析结果
    """
    if not objs:
        return None
    
    face_sizes = []
    face_ratios = []
    
    for obj in objs:
        width = obj.width
        height = obj.height
        area = obj.area
        ratio = area / (image_width * image_height)
        
        face_sizes.append({
            'width': width,
            'height': height,
            'area': area,
            'ratio': ratio,
            'aspect_ratio': width / height if height > 0 else 0
        })
        face_ratios.append(ratio)
    
    # 排序（按面积从大到小）
    face_sizes.sort(key=lambda x: x['area'], reverse=True)
    
    return {
        'face_count': len(objs),
        'sizes': face_sizes,
        'max_ratio': max(face_ratios),
        'min_ratio': min(face_ratios),
        'avg_ratio': sum(face_ratios) / len(face_ratios),
        'total_ratio': sum(face_ratios)
    }


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
        print("=== 测试Padding重试功能和人脸占比分析 ===")
        image = common.imread(test_image_path)
        
        # 正常检测
        print("\n1. 正常检测:")
        objs_normal, detection_info_normal = detect(dbface, image, use_padding_retry=False, return_face_ratio=True)
        print_face_ratio_details(detection_info_normal, "正常检测")
        
        # 人脸尺寸分析
        if objs_normal:
            analysis = analyze_face_size_distribution(objs_normal, image.shape[1], image.shape[0])
            print(f"     人脸尺寸分析:")
            for i, face_size in enumerate(analysis['sizes']):
                print(f"       人脸{i+1}: {face_size['width']:.0f}x{face_size['height']:.0f}, "
                      f"面积={face_size['area']:.0f}, 占比={face_size['ratio']:.2%}, "
                      f"宽高比={face_size['aspect_ratio']:.2f}")
        
        # 使用padding重试
        print("\n2. 使用Padding重试:")
        objs_with_padding, detection_info_with_padding = detect(dbface, image, use_padding_retry=True, return_face_ratio=True)
        print_face_ratio_details(detection_info_with_padding, "Padding重试")
        
        # 人脸尺寸分析
        if objs_with_padding:
            analysis = analyze_face_size_distribution(objs_with_padding, image.shape[1], image.shape[0])
            print(f"     人脸尺寸分析:")
            for i, face_size in enumerate(analysis['sizes']):
                print(f"       人脸{i+1}: {face_size['width']:.0f}x{face_size['height']:.0f}, "
                      f"面积={face_size['area']:.0f}, 占比={face_size['ratio']:.2%}, "
                      f"宽高比={face_size['aspect_ratio']:.2f}")
        
        # 比较结果
        print("\n3. 结果比较:")
        if len(objs_with_padding) > len(objs_normal):
            print("✓ Padding重试功能有效！检测到了更多人脸。")
            diff = len(objs_with_padding) - len(objs_normal)
            print(f"  增加了 {diff} 个人脸检测结果")
        elif len(objs_with_padding) == len(objs_normal) and len(objs_normal) > 0:
            print("= 两种方法检测结果相同。")
            # 比较人脸占比是否有变化
            normal_ratio = detection_info_normal['face_ratio_info']['face_ratio'] if detection_info_normal['face_ratio_info'] else 0
            padding_ratio = detection_info_with_padding['face_ratio_info']['face_ratio'] if detection_info_with_padding['face_ratio_info'] else 0
            ratio_diff = abs(padding_ratio - normal_ratio)
            if ratio_diff > 0.01:  # 1%的差异阈值
                print(f"  但人脸占比有变化: {normal_ratio:.2%} → {padding_ratio:.2%}")
        else:
            print("- 在这个图像上，Padding重试没有带来额外的检测结果。")
    else:
        print(f"测试图像 {test_image_path} 不存在，跳过测试。")


def batch_test_images(dbface, image_paths):
    """
    批量测试多个图像的人脸检测和占比分析
    
    Args:
        dbface: DBFace模型
        image_paths: 图像路径列表
    """
    print("=== 批量测试图像 ===")
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"\n{i+1}. 图像 {image_path} 不存在，跳过")
            continue
        
        print(f"\n{i+1}. 测试图像: {image_path}")
        image = common.imread(image_path)
        
        # 正常检测
        objs_normal, detection_info_normal = detect(dbface, image, use_padding_retry=False, return_face_ratio=True)
        
        # Padding重试检测
        objs_padding, detection_info_padding = detect(dbface, image, use_padding_retry=True, return_face_ratio=True)
        
        print(f"   正常检测: {len(objs_normal)} 个人脸")
        if detection_info_normal['face_ratio_info']:
            print(f"   正常检测人脸占比: {detection_info_normal['face_ratio_info']['face_ratio']:.2%}")
        
        print(f"   Padding检测: {len(objs_padding)} 个人脸")
        if detection_info_padding['face_ratio_info']:
            print(f"   Padding检测人脸占比: {detection_info_padding['face_ratio_info']['face_ratio']:.2%}")
        
        if len(objs_padding) > len(objs_normal):
            print(f"   ✓ Padding重试提升了检测效果 (+{len(objs_padding) - len(objs_normal)})")
        elif len(objs_padding) == len(objs_normal):
            print(f"   = 两种方法结果相同")
        else:
            print(f"   - Padding重试效果不佳")


if __name__ == "__main__":
    import os
    
    # 你可以选择运行哪个演示
    print("请选择要运行的演示:")
    print("1. 图像演示 (image_demo)")
    print("2. 摄像头演示 (camera_demo)")
    print("3. 测试Padding功能和人脸占比分析 (test_padding_feature)")
    print("4. 批量测试图像 (batch_test_images)")
    print("5. 运行所有演示")
    
    choice = input("请输入选择 (1-5, 默认为5): ").strip()
    
    if choice == "1":
        image_demo()
    elif choice == "2":
        camera_demo()
    elif choice == "3":
        test_padding_feature()
    elif choice == "4":
        # 批量测试功能
        dbface = DBFace()
        dbface.eval()
        if HAS_CUDA:
            dbface.cuda()
        dbface.load("model/dbface.pth")
        
        # 默认测试图像列表，用户可以根据需要修改
        test_images = [
            "datas/selfie.jpg",
            "datas/12_Group_Group_12_Group_Group_12_728.jpg"
        ]
        
        # 也可以让用户输入自定义路径
        print("\n使用默认测试图像还是输入自定义路径？")
        print("1. 使用默认测试图像")
        print("2. 输入自定义图像路径")
        sub_choice = input("请输入选择 (1-2, 默认为1): ").strip()
        
        if sub_choice == "2":
            print("请输入图像路径（用分号;分隔多个路径）:")
            custom_paths = input().strip()
            if custom_paths:
                test_images = [path.strip() for path in custom_paths.split(';')]
        
        batch_test_images(dbface, test_images)
    else:
        # 默认运行所有演示
        image_demo()
        camera_demo()
        test_padding_feature()
    


    