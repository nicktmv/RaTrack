#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import warnings
import logging
import torch
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt
from numba import NumbaWarning
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset_classes.track_vod_3d import TrackingDataVOD
from models import init_model
from main_utils import epoch, train_one_epoch, plot_loss_epoch, set_seed
from utils import parse_args_from_yaml


class IOStream:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    checkpoints = "checkpoints"
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    if not os.path.exists(os.path.join(checkpoints, args.exp_name)):
        os.makedirs(os.path.join(checkpoints, args.exp_name))
    if not os.path.exists(os.path.join(checkpoints, args.exp_name, "models")):
        os.makedirs(os.path.join(checkpoints, args.exp_name, "models"))


def eval_model(args, net, train_loader):
    print("Debug: net =", net)
    net.eval()
    with torch.no_grad():

        num_examples, _, _, _, seg_met, flow_met = epoch(
            args, net, train_loader, mode="eval", ep_num=99999
        )
        for key in seg_met.keys():
            seg_met[key] = seg_met[key] / num_examples
        print(seg_met)
        for key in flow_met.keys():
            flow_met[key] = flow_met[key] / num_examples
        print(flow_met)


def train(args, net, textio):
    """_summary_

    Args:
        args (_type_): _description_
        net (_type_): _description_
        textio (_type_): _description_
    """
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-10)
    scheduler = StepLR(opt, args.decay_epochs, gamma=args.decay_rate)

    best_val_loss = np.inf
    train_loss_ls = np.zeros(args.epochs)
    val_loss_ls = np.zeros(args.epochs)

    train_items_iter = {
        "Loss": [],
        "SceneFlowLoss": [],
        "SegLoss": [],
        "TrackingLoss": [],
    }

    for epoch_number in range(args.epochs):
        train_loader = DataLoader(
            TrackingDataVOD(args, args.dataset_path),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        textio.cprint(
            "==epoch: %d, learning rate: %f=="
            % (epoch_number, opt.param_groups[0]["lr"])
        )
        total_loss, loss_items = train_one_epoch(
            args, net, train_loader, opt, "train", epoch_number
        )
        train_loss_ls[epoch_number] = total_loss
        for it in loss_items:
            train_items_iter[it].extend([loss_items[it]])
            textio.cprint("%s: %f" % (it, loss_items[it]))
        textio.cprint("mean train loss: %f" % total_loss)

        if torch.cuda.device_count() > 1:
            torch.save(
                net.module.state_dict(),
                "checkpoints/%s/models/model.last.t7" % args.exp_name,
            )
            torch.save(
                net.module.state_dict(),
                "checkpoints/%s/models/model.last" % args.exp_name
                + str(epoch_number)
                + ".t7",
            )
        else:
            torch.save(
                net.state_dict(), "checkpoints/%s/models/model.last.t7" % args.exp_name
            )
            torch.save(
                net.state_dict(),
                "checkpoints/%s/models/model.last" % args.exp_name
                + str(epoch_number)
                + ".t7",
            )
        if best_val_loss >= total_loss:
            best_val_loss = total_loss
            textio.cprint("best val loss till now: %f" % total_loss)
            if torch.cuda.device_count() > 1:
                torch.save(
                    net.module.state_dict(),
                    "checkpoints/%s/models/model.best" % args.exp_name
                    + str(epoch_number)
                    + ".t7",
                )
                torch.save(
                    net.module.state_dict(),
                    "checkpoints/%s/models/model.best.t7" % args.exp_name,
                )
            else:
                torch.save(
                    net.state_dict(),
                    "checkpoints/%s/models/model.best" % args.exp_name
                    + str(epoch_number)
                    + ".t7",
                )
                torch.save(
                    net.state_dict(),
                    "checkpoints/%s/models/model.best.t7" % args.exp_name,
                )

        scheduler.step()

        plot_loss_epoch(train_items_iter, args, epoch_number)

    plt.clf()
    plt.plot(train_loss_ls[0 : int(args.epochs)], "b")
    plt.plot(val_loss_ls[0 : int(args.epochs)], "r")
    plt.legend(["train", "val"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("checkpoints/%s/loss.png" % args.exp_name, dpi=500)


def main(config_path: str):
    args = parse_args_from_yaml(config_path)

    # CUDA settings
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # deterministic results
    set_seed(args.seed)

    # init checkpoint records
    _init_(args)
    textio = IOStream("checkpoints/" + args.exp_name + "/run.log")
    textio.cprint(str(args))

    # init dataset and dataloader
    if args.dataset == "vod":
        train_loader = DataLoader(
            TrackingDataVOD(args, args.dataset_path),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
        )
    else:
        raise ValueError("Dataset not supported")

    # init the network (load or from scratch)
    net = init_model(args)

    if args.eval:
        eval_model(args, net, train_loader)
    else:
        train(args, net, textio)

    print("FINISH")


def my_main(cwd: str = ""):
    """_summary_"""
    logger = logging.getLogger("numba")
    logger.setLevel(logging.CRITICAL)

    warnings.filterwarnings("ignore", category=NumbaWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(cwd, "configs.yaml"),
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    main(args.config)


if __name__ == "__main__":
    my_main()
