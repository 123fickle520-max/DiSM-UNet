import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from unet.unet_dual_encoder_v3 import create_dual_encoder_v3


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
       
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def trainer_synapse(args, model, snapshot_path):
    logging.info(f"Start Focal Loss + Dice Loss Training. Output path: {snapshot_path}")

    # 1. 准备数据集
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

 
    dinov3_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'dinov3' in name:
            dinov3_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.SGD([
        {'params': dinov3_params, 'lr': args.base_lr * 0.1, 'initial_lr': args.base_lr * 0.1},
        {'params': other_params, 'lr': args.base_lr, 'initial_lr': args.base_lr}
    ], momentum=0.9, weight_decay=0.0001)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cuda')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"=> successfully loaded weights")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    
    focal_loss_fn = FocalLoss(gamma=2.0)
    dice_loss_fn = DiceLoss(args.num_classes)

    # 4. 训练参数设定
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = start_epoch * len(trainloader)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    scaler = torch.amp.GradScaler('cuda') 

    # 5. 正式训练循环
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(image_batch)
                
                # 计算 Focal Loss 和 Dice Loss
                loss_focal = focal_loss_fn(outputs, label_batch[:].long())
                loss_dice = dice_loss_fn(outputs, label_batch, softmax=True)
                
                
                loss = 0.5 * loss_focal + 0.5 * loss_dice

            # 梯度回传
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 学习率多项式衰减策略 
            for param_group in optimizer.param_groups:
                if 'initial_lr' in param_group:
                    param_group['lr'] = param_group['initial_lr'] * (1.0 - iter_num / max_iterations) ** 0.9

            iter_num = iter_num + 1
            
            # 记录两个部分的学习率
            writer.add_scalar('info/lr_dinov3', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/lr_others', optimizer.param_groups[1]['lr'], iter_num)
            
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_focal', loss_focal.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        logging.info(f'Epoch [{epoch_num+1}/{max_epoch}] Loss: {epoch_loss:.4f}')

       
        if (epoch_num + 1) % 5 == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num+1}.pth')
            torch.save({
                'epoch': epoch_num + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_mode_path)
            logging.info(f"Saved checkpoint: {save_mode_path}")

    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--output_dir', type=str, default='output_v3_focal', help='output dir')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logging.basicConfig(filename=os.path.join(args.output_dir, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # 初始化模型
    model = create_dual_encoder_v3(n_in=1, n_class=args.num_classes, img_size=args.img_size).cuda()
    
    # 启动训练
    trainer_synapse(args, model, args.output_dir)