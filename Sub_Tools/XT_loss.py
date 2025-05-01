import torch
import torch.nn.functional as F


def edge_loss(pred, target, w_s=5.0, w_b=0.5, lambda_weak_penalty=0.05, lambda_bce=0.5):
    """
    边缘检测损失函数，基于Dice损失，添加弱边缘惩罚和加权BCE损失，优先优化强边缘，适配归一化的target
    Args:
        pred: [B, 1, H, W], 模型输出的 logits（未经过 sigmoid）
        target: [B, 1, H, W], ground truth（归一化值：0.0, ~0.5020, 1.0）
        w_s, w_b: 强边缘、背景权重
        lambda_weak_penalty: 弱边缘惩罚权重
        lambda_bce: BCE损失权重
    Returns:
        loss: 标量，总损失
    """
    # 确保输入格式
    pred = pred.float()
    target = target.float()

    # 检查输入形状
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # sigmoid 激活
    pred_prob = torch.sigmoid(pred)

    # 转换为二值 ground truth（适配归一化值）
    strong_edge = (torch.abs(target - 1.0) < 1e-4).float()  # 强边缘：~1.0
    weak_edge = (torch.abs(target - 0.5020) < 1e-3).float()  # 弱边缘：~0.5020，收紧阈值以提高精度
    background = (torch.abs(target - 0.0) < 1e-4).float()  # 背景：~0.0

    # 验证 ground truth 值（调试用）
    unique_values = torch.unique(target)
    print(f"Unique values in target: {unique_values}")

    # Dice 损失（强边缘和背景）
    y = strong_edge  # 强边缘为 1，背景为 0，弱边缘忽略
    weight = w_s * strong_edge + w_b * background  # 弱边缘权重为 0
    intersection = (pred_prob * y * weight).sum()
    sum_pred_target = (pred_prob + y) * weight
    dice_loss = 1 - (2 * intersection + 1e-8) / (sum_pred_target.sum() + 1e-8)

    # 弱边缘惩罚：抑制弱边缘区域的预测概率
    weak_penalty = (pred_prob * weak_edge).mean()

    # 加权 BCE 损失：增强像素级分类精度
    bce_loss = F.binary_cross_entropy_with_logits(pred, strong_edge, reduction='none')
    bce_loss = (weight * bce_loss).mean()

    # 总损失：Dice + 弱边缘惩罚 + BCE
    total_loss = dice_loss + lambda_weak_penalty * weak_penalty + lambda_bce * bce_loss

    return total_loss


import torch
import torch.nn.functional as F


def edge_loss2(pred, target, w_s=5.0, w_b=0.5, lambda_weak_penalty=0.05, lambda_bce=0.5, lambda_weak_dice=0.1):
    """
    边缘检测损失函数，基于Dice损失，添加弱边缘惩罚、加权BCE损失和弱边缘Dice损失，优先优化强边缘并保留弱边缘连贯性
    Args:
        pred: [B, 1, H, W], 模型输出的 logits（未经过 sigmoid）
        target: [B, 1, H, W], ground truth（归一化值：0.0, ~0.5020, 1.0）
        w_s, w_b: 强边缘、背景权重
        lambda_weak_penalty: 弱边缘惩罚权重
        lambda_bce: BCE损失权重
        lambda_weak_dice: 弱边缘Dice损失权重
    Returns:
        loss: 标量，总损失
    """
    # 确保输入格式
    pred = pred.float()
    target = target.float()

    # 检查输入形状
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # sigmoid 激活
    pred_prob = torch.sigmoid(pred)

    # 转换为二值 ground truth（适配归一化值）
    strong_edge = (torch.abs(target - 1.0) < 1e-4).float()  # 强边缘：~1.0
    weak_edge = (torch.abs(target - 0.5020) < 1e-3).float()  # 弱边缘：~0.5020
    background = (torch.abs(target - 0.0) < 1e-4).float()  # 背景：~0.0

    # 验证 ground truth 值（调试用）
    unique_values = torch.unique(target)
    print(f"Unique values in target: {unique_values}")

    # Dice 损失（强边缘和背景）
    y = strong_edge  # 强边缘为 1，背景为 0，弱边缘忽略
    weight = w_s * strong_edge + w_b * background  # 弱边缘权重为 0
    intersection = (pred_prob * y * weight).sum()
    sum_pred_target = (pred_prob + y) * weight
    dice_loss = 1 - (2 * intersection + 1e-8) / (sum_pred_target.sum() + 1e-8)

    # 弱边缘 Dice 损失：优化弱边缘的预测
    y_weak = weak_edge
    weak_intersection = (pred_prob * y_weak).sum()
    weak_sum = (pred_prob + y_weak).sum()
    weak_dice_loss = 1 - (2 * weak_intersection + 1e-8) / (weak_sum + 1e-8)

    # 改进的弱边缘惩罚：鼓励弱边缘预测概率接近 0.5
    weak_target_prob = 0.5  # 弱边缘的目标概率
    weak_penalty = ((pred_prob - weak_target_prob).abs() * weak_edge).mean()

    # 加权 BCE 损失：弱边缘的 ground truth 设为 0.5
    bce_target = strong_edge * 1.0 + weak_edge * 0.5  # 强边缘=1.0，弱边缘=0.5，背景=0.0
    bce_loss = F.binary_cross_entropy_with_logits(pred, bce_target, reduction='none')
    bce_loss = (weight * bce_loss).mean()  # 使用与 Dice 相同的权重

    # 总损失：Dice + 弱边缘Dice + 弱边缘惩罚 + BCE
    total_loss = dice_loss + lambda_weak_dice * weak_dice_loss + lambda_weak_penalty * weak_penalty + lambda_bce * bce_loss

    return total_loss


import torch
import torch.nn.functional as F


def edge_loss3(pred, target, w_s=5.0, w_b=0.5, w_w=0.1, lambda_weak=0.05, lambda_bce=0.5, lambda_sharp=0.01):
    """
    边缘检测损失函数，基于Dice损失，添加弱边缘条件激励、加权BCE损失和局部锐利约束，
    优先优化强边缘，促进弱边缘连贯性，抑制重复预测，适配归一化的target
    Args:
        pred: [B, 1, H, W], 模型输出的 logits（未经过 sigmoid）
        target: [B, 1, H, W], ground truth（归一化值：0.0, ~0.5020, 1.0）
        w_s, w_b, w_w: 强边缘、背景、弱边缘权重
        lambda_weak: 弱边缘条件激励权重
        lambda_bce: BCE损失权重
        lambda_sharp: 局部锐利约束权重
    Returns:
        loss: 标量，总损失
    """
    # 确保输入格式
    pred = pred.float()
    target = target.float()

    # 检查输入形状
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # sigmoid 激活
    pred_prob = torch.sigmoid(pred)

    # 转换为二值 ground truth（适配归一化值）
    strong_edge = (torch.abs(target - 1.0) < 1e-4).float()  # 强边缘：~1.0
    weak_edge = (torch.abs(target - 0.5020) < 1e-3).float()  # 弱边缘：~0.5020
    background = (torch.abs(target - 0.0) < 1e-4).float()  # 背景：~0.0

    # 验证 ground truth 值（调试用）
    # unique_values = torch.unique(target)
    # print(f"Unique values in target: {unique_values}")

    # Dice 损失（强边缘和背景）
    y = strong_edge  # 强边缘为 1，背景为 0，弱边缘忽略
    weight = w_s * strong_edge + w_b * background  # 弱边缘权重为 0
    intersection = (pred_prob * y * weight).sum()
    sum_pred_target = (pred_prob + y) * weight
    dice_loss = 1 - (2 * intersection + 1e-8) / (sum_pred_target.sum() + 1e-8)

    # 弱边缘条件激励
    # 使用 3x3 卷积核计算邻域边缘密度
    kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9.0  # 均值卷积
    edge_density = F.conv2d(pred_prob * (strong_edge + weak_edge), kernel, padding=1)
    # 低密度区域（孤立弱边缘）鼓励预测，高密度区域（已有边缘）抑制预测
    weak_incentive = ((1.0 - edge_density) * weak_edge * pred_prob).mean()  # 激励孤立弱边缘
    weak_suppression = (edge_density * weak_edge * pred_prob).mean()  # 抑制重复预测

    # 加权 BCE 损失（包含弱边缘）
    bce_weight = w_s * strong_edge + w_b * background + w_w * weak_edge  # 弱边缘分配小权重
    bce_loss = F.binary_cross_entropy_with_logits(pred, strong_edge + weak_edge, reduction='none')
    bce_loss = (bce_weight * bce_loss).mean()

    # 局部锐利约束（仅针对弱边缘区域）
    grad_x = torch.abs(pred_prob[:, :, :, :-1] - pred_prob[:, :, :, 1:]) * weak_edge[:, :, :, :-1]
    grad_y = torch.abs(pred_prob[:, :, :-1, :] - pred_prob[:, :, 1:, :]) * weak_edge[:, :, :-1, :]
    sharp_loss = -(grad_x.mean() + grad_y.mean())  # 负梯度鼓励锐利边缘

    # 总损失
    total_loss = (
            dice_loss
            + lambda_weak * (weak_incentive - 0.5 * weak_suppression)  # 平衡激励和抑制
            + lambda_bce * bce_loss
            + lambda_sharp * sharp_loss
    )

    return total_loss