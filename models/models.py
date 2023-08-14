#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
from re import M
import yaml
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader

from dataloader.datasets import KittiSequentialDataset
from models.loss import MOSLoss
from models.metrics import ClassificationMetrics
from models.backbones_3d.voxel_generate import VoxelGenerate
from models.backbones_2d.mean_vfe import MeanVFE
from models.backbones_3d.spconv_unet import UNetV2 
from models.backbones_3d.motionnet import MotionNet
from models.post_process import post_processing


class InsMOSNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.poses = (
            self.hparams["DATA"]["POSES"].split(".")[0]
            if self.hparams["DATA"]["TRANSFORM"]
            else "no_poses"
        )
        self.cfg = hparams
        self.id = self.hparams["EXPERIMENT"]["ID"]
        self.dt_prediction = self.hparams["MODEL"]["DELTA_T_PREDICTION"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.n_past_steps = hparams["MODEL"]["N_PAST_STEPS"]

        self.batch_size = hparams["TRAIN"]["BATCH_SIZE"]

        self.semantic_config = yaml.safe_load(open(hparams["DATA"]["SEMANTIC_CONFIG_FILE"]))
        self.n_mos_classes = len(self.semantic_config["learning_map_inv"])
        self.ignore_index = [
            key for key, ignore in self.semantic_config["learning_ignore"].items() if ignore
        ]
        self.model = InsMOS_Model(hparams, self.n_mos_classes,self.ignore_index)

        self.ClassificationMetrics = ClassificationMetrics(self.n_mos_classes, self.ignore_index)
        # self.num_epoch_end = 0

    def forward(self, batch_data,Model_mode):
        out = self.model(batch_data,Model_mode)
        return out

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        Model_mode = 'train'

        # loss : total loss
        # train_loss_dict: L_motion, L_cls, L_reg, L_mos --- all loss
        # gt_mos_label_list: truth mos label
        # preb_mos_lable_list: predicted by insmos
        loss,train_loss_dict,gt_mos_label_list,preb_mos_lable_list = self.forward(batch,Model_mode)

        cls_loss = 0.0
        box_loss =0.0
        mos_loss = 0.0
        motion_loss = 0.0
        for i in range(0,len(train_loss_dict)):
            cls_loss = cls_loss + train_loss_dict[i]["rpn_loss_cls"]
            box_loss = box_loss + train_loss_dict[i]["rpn_loss_loc"]
            mos_loss  = mos_loss + train_loss_dict[i]["loss_mos"]
            motion_loss = motion_loss + train_loss_dict[i]["loss_motion_encoder"]
        cls_loss = cls_loss/len(train_loss_dict)
        box_loss = box_loss/len(train_loss_dict)
        mos_loss = mos_loss/len(train_loss_dict)
        motion_loss = motion_loss/len(train_loss_dict)

        self.log("train_loss", loss.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("cls_loss", cls_loss, on_step=True,batch_size=self.batch_size)
        self.log("box_loss", box_loss, on_step=True,batch_size=self.batch_size)
        self.log("mos_loss", mos_loss, on_step=True,batch_size=self.batch_size)
        self.log("motion_loss", motion_loss, on_step=True,batch_size=self.batch_size)

        batch_mos_label = torch.cat(gt_mos_label_list,dim=0)
        batch_mos_preb = torch.cat(preb_mos_lable_list,dim=0)

        # for calculating iou
        confusion_matrix = (
                self.get_step_confusion_matrix(batch_mos_preb, batch_mos_label).detach().cpu()
            )
        torch.cuda.empty_cache()
        return {"loss": loss,"confusion_matrix": confusion_matrix}

    def training_epoch_end(self, training_step_outputs):

        list_dict_confusion_matrix = [
            output["confusion_matrix"] for output in training_step_outputs
        ]

        agg_confusion_matrix = torch.zeros(self.n_mos_classes, self.n_mos_classes)
        for dict_confusion_matrix in list_dict_confusion_matrix:
            agg_confusion_matrix = agg_confusion_matrix.add(dict_confusion_matrix)
        iou = self.ClassificationMetrics.getIoU(agg_confusion_matrix)

        # save iou to tensorboard 
        self.log("train_mos_iou_step", iou[2].item(),batch_size=self.batch_size)
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):

        Model_mode = 'eval'
        metric  = {
            'batch_gt_num': 0,
        }
        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            metric['batch_recall_roi_%s' % str(cur_thresh)] = 0
            metric['batch_recall_rcnn_%s' % str(cur_thresh)] = 0

        preb_dict_list, recall_dict_list,gt_mos_label_list,preb_mos_lable_list,val_loss,val_motion_loss  = self.forward(batch,Model_mode)

        for i in range(0,len(recall_dict_list)):
            for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
                metric['batch_recall_roi_%s' % str(cur_thresh)] += recall_dict_list[i].get('roi_%s' % str(cur_thresh), 0)
                metric['batch_recall_rcnn_%s' % str(cur_thresh)] += recall_dict_list[i].get('rcnn_%s' % str(cur_thresh), 0)
            metric['batch_gt_num'] += recall_dict_list[i].get('gt',0)
    
        batch_mos_label = torch.cat(gt_mos_label_list,dim=0)
        batch_mos_preb = torch.cat(preb_mos_lable_list,dim=0)

        confusion_matrix = (
                self.get_step_confusion_matrix(batch_mos_preb, batch_mos_label).detach().cpu()
            )
        metric["confusion_matrix"] = confusion_matrix

        self.log("val_mos_loss", val_loss, on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("val_motion_loss", val_motion_loss, on_step=True,on_epoch=True,batch_size=self.batch_size)

        torch.cuda.empty_cache()
        return metric

    def validation_epoch_end(self, validation_step_outputs):

        # calculating bounding box iou
        val_metric ={
            'gt_num': 0,
        }
        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            val_metric['recall_roi_%s' % str(cur_thresh)] = 0
            val_metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        agg_confusion_matrix = torch.zeros(self.n_mos_classes, self.n_mos_classes)
        for batch_metric in validation_step_outputs:
            for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
                val_metric['recall_roi_%s' % str(cur_thresh)] += batch_metric.get('batch_recall_roi_%s' % str(cur_thresh), 0)
                val_metric['recall_rcnn_%s' % str(cur_thresh)] += batch_metric.get('batch_recall_rcnn_%s' % str(cur_thresh), 0)
            val_metric['gt_num'] += batch_metric.get('batch_gt_num', 0)

            agg_confusion_matrix = agg_confusion_matrix.add(batch_metric["confusion_matrix"])

        # calculate mos segmentation iou
        iou = self.ClassificationMetrics.getIoU(agg_confusion_matrix)
        self.log("val_mos_iou_step", iou[2].item(),batch_size=self.batch_size)
        
        gt_num_cnt = val_metric['gt_num']
        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            cur_roi_recall = val_metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = val_metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            self.log('recall_roi_%s' % str(int(cur_thresh*10)), cur_roi_recall,batch_size=self.batch_size)
            self.log('recall_rcnn_%s' % str(int(cur_thresh*10)), cur_rcnn_recall,batch_size=self.batch_size)

        torch.cuda.empty_cache()

    def get_step_confusion_matrix(self, out, past_labels):
        confusion_matrix = self.ClassificationMetrics.compute_confusion_matrix(
            out, past_labels
        )
        return confusion_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay 
        )
        return [optimizer], [scheduler]
    
    # data
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""

        ########## Point dataset splits
        train_set = KittiSequentialDataset(self.cfg, split="train")

        val_set = KittiSequentialDataset(self.cfg, split="val")

        test_set = KittiSequentialDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def collate_fn(batch):
        list_data_dict = [item for item in batch]
        return list_data_dict


#######################################
# Modles
#######################################

class InsMOS_Model(nn.Module):
    def __init__(self, cfg: dict, n_mos_classes: int,ignore_index):
        super().__init__()

        self.dt_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.post_process = cfg["MODEL"]["POST_PROCESSING"]
        self.num_class = cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"]

        self.point_cloud_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        self.voxel_size = cfg["DATA"]['VOXEL_SIZE']  
        self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size) 
        self.grid_size= np.round(self.grid_size).astype(np.int64) 

        self.n_past_step = cfg["MODEL"]["N_PAST_STEPS"]
        self.mos_class = n_mos_classes      # unlabeled static moving

        in_cahnnel = len(cfg["MODEL"]["POINT_FEATURE_ENCODING"]["src_feature_list"]) + 3 # 3 represents motion features---[unlabeled, static, moving]

        self.voxel_generate = VoxelGenerate(self.voxel_size,self.point_cloud_range,100000,5,in_cahnnel) 

        self.vfe = MeanVFE(cfg["MODEL"]["VFE"],in_cahnnel)
        self.unet = UNetV2(cfg,in_cahnnel,self.grid_size,self.voxel_size,self.point_cloud_range,self.mos_class)  # instance detection and upsample fusion

        self.motion_encoder = MotionNet(self.dt_prediction,self.voxel_size,self.mos_class)
        self.MOSLoss = MOSLoss(self.mos_class, ignore_index)
        self.use_motion_loss = cfg["MODEL"]["USE_MOTION_LOSS"]
        

    def forward(self, list_batch_dict,Model_mode):
        if Model_mode == 'train':
            loss = torch.tensor([0.], device=list_batch_dict[0]["past_labels"][-1].device)
            train_loss_dict = []

        elif  Model_mode == 'eval' :
            val_loss = torch.tensor([0.], device=list_batch_dict[0]["past_labels"][-1].device)
            val_motion_loss = torch.tensor([0.], device=list_batch_dict[0]["past_labels"][-1].device)
            preb_dict_list = []
            recall_dict_list =[]

        elif Model_mode == 'test':
            preb_dict_list = []
            recall_dict_list =[]        
        gt_mos_label_list =[]
        preb_mos_lable_list = []
        for i in range(0,len(list_batch_dict)):
            batch_dict = list_batch_dict[i]

            # motion encoding
            batch_dict = self.motion_encoder(batch_dict)

            if self.use_motion_loss == False:
                batch_dict["current_motion_feature"] = batch_dict["current_motion_feature"][:,:3]

            if Model_mode !='test':
                gt_mos_labels = batch_dict["past_labels"][-1]
                loss_motion_encoder = self.MOSLoss.compute_loss(batch_dict["current_motion_feature"],gt_mos_labels)

            # instance detection input : 3D voxel
            batch_dict = self.voxel_generate(batch_dict)
            batch_dict = self.vfe(batch_dict)

            if Model_mode =='train':
                # instance detection and upsample fusion
                (loss_rpn, tb_dict),point_seg_feature = self.unet(batch_dict,Model_mode)
                loss_mos = self.MOSLoss.compute_loss(point_seg_feature, gt_mos_labels)

                if self.use_motion_loss:
                    loss = loss + loss_rpn + loss_mos + loss_motion_encoder # loss func: loss_rpn{loss_cls,loss_reg} + loss_mos + loss_motion
                else:
                    loss = loss + loss_rpn + loss_mos

                tb_dict = {
                    'loss_mos': loss_mos.item(),
                    'loss_motion_encoder': loss_motion_encoder.item(),
                    **tb_dict
                    }
                train_loss_dict.append(tb_dict) # loss results
                gt_mos_label_list.append(gt_mos_labels) # true mos labels 
                preb_mos_lable_list.append(point_seg_feature) # preb mos labels

            elif Model_mode =='eval':
                point_seg_feature,pred_dicts, recall_dicts= self.unet(batch_dict,Model_mode)

                val_loss = val_loss + self.MOSLoss.compute_loss(point_seg_feature, gt_mos_labels)
                val_motion_loss = val_motion_loss + loss_motion_encoder.item()

                preb_dict_list.append(pred_dicts) # preb bounding box 
                recall_dict_list.append(recall_dicts) # recall of bounding box
                gt_mos_label_list.append(gt_mos_labels) 
                preb_mos_lable_list.append(point_seg_feature)

            elif Model_mode =='test':
                point_seg_feature,pred_dicts, recall_dicts= self.unet(batch_dict,Model_mode)
                preb_dict_list.append(pred_dicts)
                recall_dict_list.append(recall_dicts)
                preb_mos_lable_list.append(point_seg_feature) 
            
        if Model_mode =='train':   
            loss = loss/len(list_batch_dict)
            return loss ,train_loss_dict,gt_mos_label_list,preb_mos_lable_list

        elif Model_mode =='eval' :
            val_loss =val_loss/len(list_batch_dict)
            val_motion_loss = val_motion_loss/len(list_batch_dict)
            return preb_dict_list, recall_dict_list ,gt_mos_label_list,preb_mos_lable_list,val_loss.item(),val_motion_loss

        elif Model_mode =='test':
            return preb_dict_list, recall_dict_list ,preb_mos_lable_list

