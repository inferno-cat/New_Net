import torch
import logging
import os.path as osp
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os


def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total


def send_email(receiver: str):
    content = "Your Training is Complete!"

    msg = MIMEMultipart()
    msg.attach(MIMEText(content, "plain", "utf-8"))
    msg["Subject"] = "Training Done"
    msg["From"] = "2832941153@qq.com"
    msg["To"] = receiver

    smtp = smtplib.SMTP_SSL("smtp.qq.com", 465)
    smtp.login("2832941153@qq.com", "2003wumen")
    smtp.sendmail("2832941153@qq.com", receiver, msg.as_string())
    smtp.quit()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def save_checkpoint(state, path="./checkpoint.pth"):
    """
    Save current state as checkpoint
    :param state: Model state.
    :param path:  Path of checkpoint file.
    """
    torch.save(state, path)



# def load_checkpoint(net, opt=None, path="./checkpoint.pth"):
#     """
#     Load previous pre-trained checkpoint.
#     :param net:  Network instance.
#     :param opt:  Optimizer instance.
#     :param path: Path of checkpoint file.
#     :return:     Checkpoint epoch number.
#     """
#     if osp.isfile(path):
#         print("=> Loading checkpoint {}...".format(path))
#         checkpoint = torch.load(path)
#         # tag
#         # checkpoint = torch.load(path, weights_only=False)
#
#         net.load_state_dict(checkpoint["model"])
#
#
#
#         if opt is not None:
#             opt.load_state_dict(checkpoint["opt"])
#         return checkpoint["epoch"]
#     else:
#         raise ValueError("=> No checkpoint found at {}.".format(path))

import os.path as osp
import torch
import torch.nn as nn

def load_checkpoint(net, opt=None, path="./checkpoint.pth"):
    """
    加载预训练检查点，兼容 nn.DataParallel。
    :param net:  网络实例
    :param opt:  优化器实例（可选）
    :param path: 检查点文件路径
    :return:     检查点的轮次号
    """
    if osp.isfile(path):
        print(f"=> 加载检查点 {path}...")
        # 加载检查点到当前设备（GPU 或 CPU）
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # 获取模型参数
        state_dict = checkpoint["model"]
        model_dict = net.state_dict()

        # 处理 nn.DataParallel 的键名兼容性
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果模型是 DataParallel，但键没有 'module.'，添加前缀
            if isinstance(net, nn.DataParallel) and not k.startswith("module."):
                k = f"module.{k}"
            # 如果模型不是 DataParallel，但键有 'module.'，移除前缀
            elif not isinstance(net, nn.DataParallel) and k.startswith("module."):
                k = k.replace("module.", "")
            new_state_dict[k] = v

        # 加载调整后的参数
        try:
            model_dict.update(new_state_dict)
            net.load_state_dict(model_dict, strict=True)
        except RuntimeError as e:
            print(f"加载参数出错: {e}")
            print("尝试使用 strict=False 加载...")
            net.load_state_dict(new_state_dict, strict=False)

        if opt is not None:
            opt.load_state_dict(checkpoint["opt"])
        return checkpoint["epoch"]
    else:
        raise ValueError(f"=> 在 {path} 未找到检查点。")



def convert_model_to_standard_conv(model):
    if isinstance(model, torch.nn.DataParallel):
        model.module.convert_to_standard_conv()
    else:
        model.convert_to_standard_conv()


def save_cpdc(model, path):
    convert_model_to_standard_conv(model)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
