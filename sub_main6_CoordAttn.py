import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from utils import get_logger
from datetime import datetime
# from dataset import BsdsDataset, NyudDataset, VOCDataset
from sub_dataset import BsdsDataset, NyudDataset, VOCDataset
# from tester import test_bsds , test_nyud
# from trainer import train_bsds, train_nyud, train_voc
from sub_tester import test_bsds , test_nyud
from sub_trainer import train_bsds, train_nyud, train_voc
# from model.model_eff_b7 import MyModel
# from pdc_attention_network import PDCNet
# from loss import Loss
# from utils import load_checkpoint, save_checkpoint, send_email, get_model_parm_nums, save_cpdc
from sub_loss import Loss
from sub_utils import load_checkpoint, save_checkpoint, send_email, get_model_parm_nums, save_cpdc
from torch.utils.data import DataLoader

# from Sub_Tools.ET_pdc_network import PDCNet
from sub_net6_CoordAttn import PDCNet
from sub_utils import get_logger

# 1.参数定义
def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Training/Testing")
    parser.add_argument("--seed", default=42, type=int, help="seed for initialization")
    parser.add_argument("--test", default=False, help="Only test the model", action="store_true")
    parser.add_argument("--ms", default=False, help="Multi-scale test the model", action="store_true")
    parser.add_argument("--train_batch_size", default=4, type=int, metavar="N", help="training batch size")
    parser.add_argument("--test_batch_size", default=1, type=int, metavar="N", help="testing batch size")
    parser.add_argument("--num_workers", default=1, type=int, metavar="N", help="number of workers")
    parser.add_argument("--sampler_num", default=20000, type=int, metavar="N", help="sampler num")
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "--lr", "--learning_rate", default=1e-4, type=float, metavar="LR", help="initial learning rate", dest="lr"
    )
    parser.add_argument(
        "--lr_stepsize", default=5, type=int, metavar="N", help="decay lr by a factor every lr_stepsize epochs"
    )
    parser.add_argument("--lr_gamma", default=0.1, type=float, metavar="F", help="learning rate decay factor (gamma)")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="F", help="momentum")
    parser.add_argument(
        "--weight_decay", "--wd", default=0.0005, type=float, metavar="F", help="weight decay (default: 0.0005)"
    )
    parser.add_argument("--print_freq", "-p", default=500, type=int, metavar="N", help="print frequency (default: 500)")
    parser.add_argument(
        "--resume", default=r"", type=str, metavar="PATH", help="path to latest checkpoint (default: none)"
    )
    parser.add_argument("--store_folder", default="./output", type=str, metavar="PATH", help="path to store folder")
    # parser.add_argument(
    #     "--dataset", default="./data/BSDS500_flipped_rotated", type=str, metavar="PATH", help="path to dataset"
    # )
    #tag
    parser.add_argument(
        "--dataset", default=r"D:\rawcode\data\BSDS500_flipped_rotated_pad", type=str, metavar="PATH", help="path to dataset"
    )
    parser.add_argument(
        "--optimizer_method", default="Adam", type=str, metavar="OPT", help="optimizer method (default: Adam)"
    )
    parser.add_argument(
        "--loss_method",
        default="HFL",
        type=str,
        metavar="LOSS",
        help="loss method (default: Weighted Cross Entropy Loss)",
    )
    parser.add_argument("--amp", default=False, help="Use mixed precision training", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    # 设置随机种子
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('device:', device)

    import subprocess
    import sys

    # 在导入PyTorch之前设置只使用GPU 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    def check_gpu_memory(gpu_id=4, max_usage_ratio=0.1):
        """
        检查指定GPU的显存占用率。
        参数：
            gpu_id: 要检查的GPU编号
            max_usage_ratio: 最大允许的显存占用率（例如0.1表示10%）
        返回：
            True如果GPU空闲（显存占用率低于max_usage_ratio），否则False
        """
        try:
            # 调用nvidia-smi获取GPU显存信息
            result = subprocess.check_output(
                'nvidia-smi --query-gpu=memory.used,memory.total --format=csv',
                shell=True
            ).decode('utf-8')

            # 解析nvidia-smi输出
            lines = result.strip().split('\n')[1:]  # 跳过表头
            if gpu_id >= len(lines):
                print(f"Error: GPU {gpu_id} does not exist.")
                return False

            # 获取指定GPU的显存使用情况
            used, total = map(lambda x: int(x.split()[0]), lines[gpu_id].split(','))
            usage_ratio = used / total
            print(f"GPU {gpu_id} memory usage: {used}MiB/{total}MiB ({usage_ratio:.2%})")

            # 检查显存占用率
            if usage_ratio > max_usage_ratio:
                print(f"Error: GPU {gpu_id} is in use (memory usage > {max_usage_ratio:.0%}).")
                return False
            return True
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            return False

    # 检查GPU 4是否空闲
    if not check_gpu_memory(gpu_id=4, max_usage_ratio=0.1):
        print("Exiting program due to GPU 4 being in use.")
        sys.exit(1)

    # 设置设备
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    torch.cuda.set_device(4)
    torch.cuda.empty_cache()

    # 验证是否使用正确的GPU
    if device.type == "cuda":
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"Using GPU {current_gpu} ({gpu_name})")
    else:
        print("Error: No GPU available, falling back to CPU.")
        sys.exit(1)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    store_dir = os.path.join(current_dir, args.store_folder)
    print('current_dir:', current_dir)
    print('store_dir:', store_dir)
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger = get_logger(os.path.join(store_dir, "log-{}.txt".format(now_str)))

    # 1.数据
    test_dataset = BsdsDataset(dataset_path=args.dataset, flag="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,

    )

    # 2.定义网络
    model = PDCNet(16).to(device)
    # model = MyModel().to(device)
    # model = nn.DataParallel(model)
    logger.info("The number of parameters: {:.2f}M".format(get_model_parm_nums(model)))
    logger.info(args)

    # 3.定义优化器
    opt = None
    if args.optimizer_method == "Adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_method == "SGD":
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_method == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4.定义学习策略
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=opt, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    # 5.损失函数
    loss = Loss(args.loss_method)
    loss.to(device)

    # 6.训练
    if args.resume:
        load_checkpoint(model, path=args.resume)


    if args.amp:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    import time
    time.sleep(1000) 
    if args.test is True:
        test_bsds(
            test_loader,
            model,
            save_dir=os.path.join(store_dir, "test"),
            logger=logger,
            device=device,
            multi_scale=args.ms,
        )
    else:
        train_epoch_losses = []
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            if epoch == 0:
                logger.info("Initial test...")
                # test_bsds(
                #     test_loader,
                #     model,
                #     save_dir=os.path.join(store_dir, "initial_test"),
                #     logger=logger,
                #     device=device,
                #     multi_scale=args.ms,
                # )
            train_dataset = BsdsDataset(dataset_path=args.dataset, flag="train", sub_sample=args.sampler_num)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            # 数据集采样
            train_epoch_loss = train_bsds(
                train_loader,
                model,
                opt,
                lr_scheduler,
                args.print_freq,
                args.epochs,
                epoch,
                save_dir=os.path.join(store_dir, "epoch-{}-train".format(epoch + 1)),
                logger=logger,
                device=device,
                loss=loss,
                AT_gradient_accumulation_steps=-1,
            )

            test_bsds(
                test_loader,
                model,
                save_dir=os.path.join(store_dir, "epoch-{}-test".format(epoch + 1)),
                logger=logger,
                device=device,
                multi_scale=args.ms,
            )

            lr_scheduler.step()

            # 保存模型
            # save_checkpoint(
            #     state={"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
            #     path=os.path.join(store_dir, "epoch-{}-ckpt.pt".format(epoch + 1)),
            # )

            # t1 = time.time()
            save_cpdc(model, path=os.path.join(store_dir, "epoch-{}-ckpt.pt".format(epoch + 1)))
            save_checkpoint(model, opt, epoch + 1, path=os.path.join(store_dir, 'checkpoints', "checkpoint_epoch-{}-ckpt.pt".format(epoch + 1)))

            # 收集每个epoch的loss
            train_epoch_losses.append(train_epoch_loss)

    # # 训练完毕，发送邮件提醒
    # send_email("2832941153@qq.com")


if __name__ == "__main__":
    main()
