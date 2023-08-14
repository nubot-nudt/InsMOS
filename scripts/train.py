#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import click
import yaml
import os
import sys
sys.path.append('.')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import models.models as models
from easydict import EasyDict
data_cfg = EasyDict()

    
@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="./config/config.yaml",
)
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    help="path to checkpoint file (.ckpt) to resume training.",
    default=None,
)
def main(config, weights, checkpoint):

    if checkpoint:
        cfg = torch.load(checkpoint)["hyper_parameters"]
    else:
        cfg = yaml.safe_load(open(config))


    model = models.InsMOSNet(cfg)
    if weights is None:
        model = models.InsMOSNet(cfg)
    else:
        model = models.InsMOSNet.load_from_checkpoint(weights, hparams=cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        save_top_k = 2,
        monitor="val_mos_iou_step",
        filename=cfg["EXPERIMENT"]["ID"] + "_{epoch:03d}_{val_mos_iou_step:.3f}",
        mode="max",
        save_last=True,
    )

    # Logger
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        log_dir, name=cfg["EXPERIMENT"]["ID"], default_hp_metric=False
    )
    # Setup trainer
    trainer = Trainer(
        # gpus=1,               # for sigle GPU
        accelerator="gpu",  # for Multi gpu
        devices=[0,1,2,3],  # for Multi GPU
        strategy="ddp",
        logger=tb_logger,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        accumulate_grad_batches=cfg["TRAIN"]["ACC_BATCHES"],
        callbacks=[lr_monitor, checkpoint_saver],
    )
    # Train!
    trainer.fit(model, ckpt_path=checkpoint)

if __name__ == "__main__":
    
    main()
