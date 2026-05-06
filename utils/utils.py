import torch
import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk

from thop import profile
from thop import clever_format


# =========================
# 基础工具函数
# =========================

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


# =========================
# 统计类
# =========================

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[max(len(self.losses) - self.num, 0):])
        )


# =========================
# FLOPs / Params
# =========================

def CalParams(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]')
    print('FLOPs:', flops)
    print('Params:', params)


# =========================
# One-hot
# =========================

def one_hot_encoder(input_tensor, dataset, n_classes=None):
    tensor_list = []
    if dataset == 'MMWHS':
        label_map = [0, 205, 420, 500, 550, 600, 820, 850]
        for i in label_map:
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
    return torch.cat(tensor_list, dim=1).float()


# =========================
# Dice Loss（训练用）
# =========================

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


# =========================
# Metric
# =========================

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return (
            metric.binary.dc(pred, gt),
            metric.binary.hd95(pred, gt),
            metric.binary.jc(pred, gt),
            metric.binary.assd(pred, gt)
        )
    elif pred.sum() > 0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def calculate_metric_percase_dice(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt), 0, 0, 0
    elif pred.sum() > 0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt)
    elif pred.sum() > 0:
        return 1
    else:
        return 0


# =========================
# 测试（Dice-only）
# =========================

def test_single_volume_dice(
    image, label, net, classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1
):
    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()
    device = next(net.parameters()).device

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind]
        x, y = slice.shape
        if (x, y) != tuple(patch_size):
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            if isinstance(outputs, list):
                outputs = sum(outputs)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

        if (x, y) != tuple(patch_size):
            out = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

        prediction[ind] = out

    metric_list = [
        calculate_metric_percase_dice(prediction == i, label == i)
        for i in range(1, classes)
    ]

    if test_save_path is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(prediction.astype(np.float32)),
            f"{test_save_path}/{case}_pred.nii.gz"
        )

    return metric_list


# =========================
# 测试（完整指标）
# =========================

def test_single_volume(
    image, label, net, classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1
):
    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()
    device = next(net.parameters()).device

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind]
        x, y = slice.shape
        if (x, y) != tuple(patch_size):
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            if isinstance(outputs, list):
                outputs = sum(outputs)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

        if (x, y) != tuple(patch_size):
            out = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

        prediction[ind] = out

    metric_list = [
        calculate_metric_percase(prediction == i, label == i)
        for i in range(1, classes)
    ]

    if test_save_path is not None:
        sitk.WriteImage(
            sitk.GetImageFromArray(prediction.astype(np.float32)),
            f"{test_save_path}/{case}_pred.nii.gz"
        )

    return metric_list


# =========================
# 验证
# =========================

def val_single_volume(
    image, label, net, classes,
    patch_size=[256, 256]
):
    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()
    device = next(net.parameters()).device

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind]
        x, y = slice.shape
        if (x, y) != tuple(patch_size):
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            if isinstance(outputs, list):
                outputs = sum(outputs)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

        if (x, y) != tuple(patch_size):
            out = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

        prediction[ind] = out

    return [
        calculate_dice_percase(prediction == i, label == i)
        for i in range(1, classes)
    ]
