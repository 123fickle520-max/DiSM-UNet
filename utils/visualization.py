import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- 配置区域 ---
# Synapse 8器官的标准颜色定义 (RGB格式)
ORGAN_COLORS = {
    1: (35, 35, 142),    # Aorta
    2: (0, 148, 148),    # Gallbladder
    3: (205, 133, 63),   # Kidney(L)
    4: (85, 127, 47),    # Kidney(R)
    5: (186, 85, 211),   # Liver
    6: (75, 0, 130),     # Pancreas
    7: (220, 220, 220),  # Spleen
    8: (200, 200, 60)    # Stomach
}

def normalize_for_display(img_data):
    """将CT值归一化到 0-255 用于显示"""
    img_data = img_data.astype(np.float32)
    img_min = img_data.min()
    img_max = img_data.max()
    if img_max > img_min:
        img_norm = (img_data - img_min) / (img_max - img_min)
    else:
        img_norm = img_data
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return img_uint8

def create_overlay(img_gray, mask, alpha=0.6):
    """创建半透明彩色叠加图 (Numpy 纯净版)"""
    # 转为RGB
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # 1. 检查空掩膜
    if np.sum(mask) == 0:
        return img_rgb

    # 2. 生成彩色遮罩
    colored_mask = np.zeros_like(img_rgb)
    for label_idx, color in ORGAN_COLORS.items():
        colored_mask[mask == label_idx] = color
        
    foreground = mask > 0
    
    # 3. 安全检查: 如果前景为空，直接返回
    if not np.any(foreground):
        return img_rgb

    # 4. [核心修复]: 使用 Numpy 纯数学运算替代 cv2.addWeighted
    # 避免 OpenCV 在处理 (N, 3) 形状数组时的兼容性报错
    overlay_rgb = img_rgb.copy()
    
    # 取出前景区域 (转为 float 进行精确计算)
    img_part = img_rgb[foreground].astype(np.float32)
    mask_part = colored_mask[foreground].astype(np.float32)
    
    # 混合公式: 结果 = 原图 * (1-alpha) + 掩膜 * alpha
    blended = img_part * (1 - alpha) + mask_part * alpha
    
    # 填回原图 (转回 uint8)
    overlay_rgb[foreground] = blended.astype(np.uint8)
    
    return overlay_rgb

def save_global_visualization(image, label, prediction, case_name, save_dir, slice_id, mean_dice, method_name="Method", rank=None):
    """
    保存可视化结果
    """
    # 1. 数据准备
    img_uint8 = normalize_for_display(image)
    label_uint8 = label.astype(np.uint8)
    pred_uint8 = prediction.astype(np.uint8)
    
    # 2. 创建叠加图
    gt_overlay = create_overlay(img_uint8, label_uint8, alpha=0.6)
    pred_overlay = create_overlay(img_uint8, pred_uint8, alpha=0.6)
    
    # 3. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Left: Ground Truth
    axes[0].imshow(gt_overlay)
    axes[0].set_title(f"Ground Truth (Slice {slice_id})", fontsize=11, fontweight='bold', y=-0.12)
    axes[0].axis('off')
    
    # Right: Prediction
    axes[1].imshow(pred_overlay)
    axes[1].set_title(f"{method_name}\nMean Dice: {mean_dice:.4f}", fontsize=11, fontweight='bold', y=-0.18)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 4. 生成文件名
    dice_str = f"{mean_dice:.4f}"
    if rank is not None:
        file_name = f"Rank{rank:03d}_Dice{dice_str}_{case_name}_slice{slice_id:03d}.png"
    else:
        file_name = f"{case_name}_slice{slice_id:03d}_mDice{int(mean_dice*100)}.png"

    save_path = os.path.join(save_dir, file_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()