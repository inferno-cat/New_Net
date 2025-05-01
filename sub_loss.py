import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, List


# from Sub_Tools.XT_loss import edge_loss3 as XT_loss


class Loss(nn.Module):
    def __init__(self, loss_function="WCE"):
        super(Loss, self).__init__()
        self.name = loss_function

    def forward(self, preds, labels) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.name == "BCE":
            final_loss = bce_loss(preds, labels)
        elif self.name == "WCE":
            final_loss = weighted_bce_loss(preds, labels)
        elif self.name == "Dice":
            final_loss = 0.001 * bce_loss(preds, labels) + dice_loss(preds, labels)
        elif self.name == "SSIM":
            final_loss = ssim_loss(preds, labels) + 0.001 * bce_loss(preds, labels)
        elif self.name == "IOU":
            final_loss = iou_loss(preds, labels) + 0.1 * weighted_bce_loss(preds, labels)
        elif self.name == "Tversky":
            final_loss = tversky_loss(preds, labels) + 0.1 * weighted_bce_loss(preds, labels)
        elif self.name == "FTL++":
            final_loss = focal_tversky_loss_plus_plus(preds, labels)
        elif self.name == "HFL":
            final_loss = focal_tversky_loss_plus_plus(preds, labels) + 0.001 * focal_loss(preds, labels)
        elif self.name == "AW":
            final_loss = AutoWeighted_Tversky_and_WCE_and_Focal_loss()(preds, labels)
        elif self.name == "RCF":
            final_loss = RCF_WCE(preds, labels)
        elif self.name == "Grad":
            final_loss = grad_loss(preds, labels)

        # elif self.name == "XT":
        #     final_loss = XT_loss(preds, labels)
        else:
            raise NameError

        return final_loss


def bce_loss(preds, labels):
    bce_loss = F.binary_cross_entropy(preds, labels, reduction="sum")

    return bce_loss


def weighted_bce_loss(preds, labels):
    beta = 1 - torch.mean(labels)
    weights = 1 - beta + (2 * beta - 1) * labels
    wce_loss = F.binary_cross_entropy(preds, labels, weights, reduction="sum")

    return wce_loss


def RCF_WCE(preds, labels):
    n = preds.size(0)
    wce_loss = 0.0
    for i in range(n):
        # 获取当前样本
        current_pred = preds[i]
        current_label = labels[i]
        current_weight = current_label.clone()

        num_positive = torch.sum((current_label == 1).float()).float()
        num_negative = torch.sum((current_label == 0).float()).float()

        # 计算权重，使用torch.where避免原地操作
        current_weight[current_weight == 1] = 1.0 * num_negative / (num_positive + num_negative)
        current_weight[current_weight == 0] = 1.1 * num_positive / (num_positive + num_negative)
        selected_idx = current_weight != 2
        current_pred = current_pred[selected_idx]
        current_label = current_label[selected_idx]
        current_weight = current_weight[selected_idx]
        temp_loss = F.binary_cross_entropy(current_pred, current_label, weight=current_weight, reduction='sum')
        wce_loss = wce_loss + temp_loss

    return wce_loss


def dice_loss(preds, labels):
    n = preds.size(0)
    dice_loss = 0.0
    for i in range(n):
        prob_2 = torch.mul(preds[i, :, :, :], preds[i, :, :, :])
        label_2 = torch.mul(labels[i, :, :, :], labels[i, :, :, :])
        prob_label = torch.mul(preds[i, :, :, :], labels[i, :, :, :])
        sum_prob_2 = torch.sum(prob_2)
        sum_label_2 = torch.sum(label_2)
        sum_prob_label = torch.sum(prob_label)
        sum_prob_label = sum_prob_label + 0.000001
        temp_loss = (sum_prob_2 + sum_label_2) / (2 * sum_prob_label)
        if temp_loss.data.item() > 50:
            temp_loss = 50
        dice_loss = dice_loss + temp_loss

    return dice_loss


def ssim_loss(preds, labels):
    n, c, h, w = preds.shape
    pixel_total_num = h * w
    C = 0.000001
    # C1 = 0.01**2
    # C2 = 0.03**2
    ss_loss = 0.0
    for i in range(n):
        pred_mean = torch.mean(preds[i, :, :, :])
        pred_var = torch.var(preds[i, :, :, :])
        label_mean = torch.mean(labels[i, :, :, :])
        label_var = torch.var(labels[i, :, :, :])
        pred_label_var = (
            torch.abs(preds[i, :, :, :] - pred_mean) * torch.abs(labels[i, :, :, :] - label_mean)
        ).sum() / (pixel_total_num - 1)

        # temp_loss = ((torch.square(pred_mean) + torch.square(label_mean)) * (pred_var + label_var) + C) / (
        #         (2 * pred_mean * label_mean) * (2 * pred_label_var) + C)
        temp_loss = (pred_var * label_var + C) / (pred_label_var + C)
        ss_loss = ss_loss + temp_loss

    return ss_loss


def iou_loss(preds, labels):
    iou_loss = 0.0
    n = preds.shape[0]
    C = 0.000001
    for i in range(n):
        Iand = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        Ior = torch.sum(preds[i, :, :, :]) + torch.sum(labels[i, :, :, :]) - Iand

        # temp_loss = -torch.log((Iand + C) / (Ior + C))
        temp_loss = Iand / Ior
        iou_loss = iou_loss + (1 - temp_loss)

    return iou_loss


def tversky_loss(preds, labels):
    tversky_loss = 0.0
    beta = 0.7
    alpha = 1.0 - beta
    C = 0.000001
    n = preds.shape[0]
    for i in range(n):
        tp = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        fp = torch.sum(preds[i, :, :, :] * (1 - labels[i, :, :, :]))
        fn = torch.sum((1 - preds[i, :, :, :]) * labels[i, :, :, :])
        temp_loss = -torch.log((tp + C) / (tp + alpha * fp + beta * fn + C))

        tversky_loss = tversky_loss + temp_loss

    return tversky_loss


def focal_tversky_loss_plus_plus(preds, labels, gamma: float = 2, beta: float = 0.7, delta: float = 0.75):
    focal_tversky_loss = 0.0
    epsilon = 1e-7
    n = preds.shape[0]
    for i in range(n):
        tp = torch.sum(preds[i, :, :, :] * labels[i, :, :, :])
        fp = torch.sum((preds[i, :, :, :] * (1 - labels[i, :, :, :])) ** gamma)
        fn = torch.sum(((1 - preds[i, :, :, :]) * labels[i, :, :, :]) ** gamma)
        tversky = (tp + (1 - beta) * fp + beta * fn + epsilon) / (tp + epsilon)
        temp_loss = torch.pow(tversky, delta)
        if temp_loss.data.item() > 50.0:
            temp_loss = torch.clamp(temp_loss, max=50.0)

        focal_tversky_loss = focal_tversky_loss + temp_loss

    return focal_tversky_loss


def focal_loss(preds, labels, alpha: float = 0.25, gamma: float = 2, reduction: str = "sum"):
    bce_cross_entropy = F.binary_cross_entropy(preds, labels, reduction=reduction)
    pt = torch.exp(-bce_cross_entropy)
    focal_loss = alpha * ((1 - pt) ** gamma) * bce_cross_entropy

    return focal_loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        #         for param in self.parameters():
        #             print(param)
        return loss_sum


class AutoWeighted_Tversky_and_WCE_and_Focal_loss(nn.Module):
    def __init__(self):
        super(AutoWeighted_Tversky_and_WCE_and_Focal_loss, self).__init__()
        self.tversky = focal_tversky_loss_plus_plus
        self.wce = weighted_bce_loss
        self.focal = focal_loss
        self.awl = AutomaticWeightedLoss(3)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        tv_loss = self.tversky(preds, labels)
        wce_loss = self.wce(preds, labels)
        focal_loss = self.focal(preds, labels)
        result = self.awl(tv_loss, wce_loss, focal_loss)
        return result


def grad_loss(preds, original_images):
    device = preds.device

    # RGB->gray
    gray_weights = torch.tensor([0.299, 0.587, 0.114]).to(device).view(1, 3, 1, 1)
    original_gray = (original_images * gray_weights).sum(dim=1, keepdim=True)

    filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    filter_xy = torch.cat([filter_x, filter_y], dim=0)

    # grad_preds = F.conv2d(preds, filter_xy, padding=1)
    grad_labels = F.conv2d(original_gray, filter_xy, padding=1)

    return F.l1_loss(preds, grad_labels, reduction="sum")
    # return F.mse_loss(preds, grad_labels, reduction="mean")


if __name__ == "__main__":
    pass
