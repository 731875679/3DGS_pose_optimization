#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

# SSIM loss 度量两个给定图像之间的相似性。
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # 在张量的维度上添加一个维度，具体来说，在索引 1 的位置上添加一个新的维度，使得原本的一维张量变成了一个二维张量，其中一个维度的大小是 1。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # _1D_window.mm(_1D_window.t())：这一部分进行了矩阵乘法操作。首先，_1D_window 通过 .t() 被转置为列向量（一维行向量），然后对这两个向量进行矩阵乘法操作。
    # .unsqueeze(0).unsqueeze(0)：这两个操作在结果张量的两个维度上各添加一个维度。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # .contiguous() 方法用于返回一个连续的（contiguous）副本张量，确保了张量在内存中的存储是连续的，这对后续的操作是必要的。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_loss_tracking(opt, rendered_image, depth, opacity, viewpoint):
    """
    计算相机位姿更新过程中的损失。

    Args:
        opt: 训练选项或配置对象。
        rendered_image: 渲染图像。
        depth: 渲染深度图。
        opacity: 渲染不透明度图。
        viewpoint: 相机视点对象，包含原始图像和其他信息。

    Returns:
        torch.Tensor: 计算出的损失值。
    """
    # 获取原始图像
    target_image = viewpoint.original_image
    
    # 计算渲染图像与目标图像之间的均方误差（MSE）
    loss = F.mse_loss(rendered_image, target_image)
    
    # 如果需要，可以在此处添加其他损失项（例如深度或不透明度损失）
    # loss += F.mse_loss(depth, target_depth)
    # loss += F.mse_loss(opacity, target_opacity)

    return loss
