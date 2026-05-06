import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset

from unet.unet_dual_encoder_v3 import create_dual_encoder_v3
from utils.utils import calculate_metric_percase
from utils.visualization import save_global_visualization

# --- 配置 ---
SAVE_TOP_K = 3  # 每个 Case 只保存最具代表性的前3张切片

def calculate_slice_global_mean(pred_slice, label_slice, num_classes=9):
    """计算单张切片的平均 Dice (忽略背景0)"""
    dices = []
    for c in range(1, num_classes):
        if np.sum(label_slice == c) > 0:
            pred_bin = (pred_slice == c)
            label_bin = (label_slice == c)
            intersection = np.sum(pred_bin * label_bin)
            union = np.sum(pred_bin) + np.sum(label_bin)
            dice = (2.0 * intersection) / union if union > 0 else 0.0
            dices.append(dice)
    if not dices: 
        return 0.0
    return np.mean(dices)

def inference(model, test_loader, args):
    model.eval()
    
    vis_dir = os.path.join(args.output_dir, "vis_Top3_MostOrgans_HighDice")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    total_metric = np.zeros((args.num_classes - 1, 2))
    
    logging.info("-" * 80)
    logging.info(f"Visualization Strategy: 1. Most Organs -> 2. Highest Dice -> Save Top {SAVE_TOP_K}")
    logging.info(f"Saving to: {vis_dir}")
    logging.info("-" * 80)
    
    for i_batch, sampled_batch in enumerate(tqdm(test_loader, desc="Testing")):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        # 移除 batch 维度 (Synapse 测试集单例 batch)
        image = image.squeeze(0).cpu().detach().numpy()
        label = label.squeeze(0).cpu().detach().numpy()
        
        depth, h, w = image.shape
        prediction = np.zeros_like(label)

        # --- 切片级推理过程 ---
        for i in range(depth):
            device = torch.device('cpu' if (hasattr(args, 'force_cpu') and args.force_cpu) else ('cuda' if torch.cuda.is_available() else 'cpu'))

            # 扩展维度为 [B=1, C=1, H, W]
            slice_tensor = torch.from_numpy(image[i]).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # 缩放至模型所需大小
            if h != args.img_size or w != args.img_size:
                slice_tensor = F.interpolate(slice_tensor, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                outputs = model(slice_tensor)
                
                # 恢复原图大小
                if h != args.img_size or w != args.img_size:
                    outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=False)
                
                # 获取类别预测
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                prediction[i] = out.cpu().detach().numpy()

        # --- 全局体素级别指标计算 ---
        metric_i = []
        for c in range(1, args.num_classes):
            metric = calculate_metric_percase(prediction == c, label == c)
            if isinstance(metric, (tuple, list, np.ndarray)) and len(metric) > 2:
                metric = metric[:2]
            total_metric[c-1, :] += metric
            metric_i.append(metric)
        
        metric_i = np.array(metric_i)
        case_mean_dice = np.mean(metric_i[:, 0])
        case_mean_hd95 = np.mean(metric_i[:, 1])
        logging.info(f'[{i_batch+1}/{len(test_loader)}] Case: {case_name} | Mean Dice: {case_mean_dice:.4f} | Mean HD95: {case_mean_hd95:.4f}')
        
        # --- 筛选与可视化保存逻辑 ---
        slice_candidates = []
        for z in range(depth):
            unique_labels = np.unique(label[z])
            # 计算当前切片存在的真实器官数量 (排除背景0)
            num_organs = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
            
            if num_organs > 0:
                m_dice = calculate_slice_global_mean(prediction[z], label[z], args.num_classes)
                slice_candidates.append((m_dice, z, num_organs))
        
        # 排序：优先包含的器官数 (降序)，器官数相同时优先 Dice 分数 (降序)
        slice_candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
        top_k_slices = slice_candidates[:SAVE_TOP_K]
        
        for rank_idx, (score, idx, n_organs) in enumerate(top_k_slices):
            save_global_visualization(
                image[idx], label[idx], prediction[idx], 
                case_name, vis_dir, 
                slice_id=idx, 
                mean_dice=score,
                method_name=args.method_name,
                rank=(rank_idx + 1)
            )

    # --- 最终指标汇总 ---
    avg_metric = total_metric / len(test_loader)
    logging.info("=" * 80)
    for i in range(args.num_classes - 1):
        logging.info(f"Class {i+1} | Mean Dice: {avg_metric[i, 0]:.4f} | Mean HD95: {avg_metric[i, 1]:.4f}")
    logging.info("=" * 80)
    
    return np.mean(avg_metric[:, 0]), np.mean(avg_metric[:, 1])

def test_dual_encoder_v3(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'test_log_finetuned.txt')
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(message)s', 
        datefmt='%H:%M:%S', 
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    logging.info(f"Testing Model Weight: {args.checkpoint}")
    logging.info(f"Using device: {device}")

    # 1. 创建模型 
    model = create_dual_encoder_v3(n_in=1, n_class=args.num_classes, img_size=args.img_size)
    
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        return

    # 2. 加载微调后的模型权重
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 容错加载机制：允许 strict=False 防止多卡训练带来的 module. 前缀问题
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Weights loaded strictly and successfully.")
    except Exception as e:
        logging.warning(f"Strict load failed: {e}. Trying to load with strict=False...")
        # 清除 DDP 训练可能引入的 'module.' 前缀
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        logging.info("Weights loaded with strict=False.")
    
    model = model.to(device)
    
    # 3. 准备数据
    db_test = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    
    # 4. 执行推理
    mean_dice, mean_hd95 = inference(model, testloader, args)
    logging.info(f">>> Final Testing Performance | Mean Dice: {mean_dice:.4f} | Mean HD95: {mean_hd95:.4f} <<<")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str, default='data/Synapse/test_vol_h5', help='test data path')
    parser.add_argument('--dataset', type=str, default='Synapse', help='dataset name')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int, default=224, help='must match training size')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth weight file')
    parser.add_argument('--output_dir', type=str, default='test_result_v3_finetuned', help='output dir')
    
    
    parser.add_argument('--method_name', type=str, default='DiSM-UNet', help='Display name')
    parser.add_argument('--force_cpu', action='store_true', help='Force usage of CPU')

    args = parser.parse_args()
    test_dual_encoder_v3(args)