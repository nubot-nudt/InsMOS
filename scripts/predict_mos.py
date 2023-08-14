#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from pytorch_lightning import Trainer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import sys
sys.path.append('.')

import copy
import numpy as np
from pathlib import Path
import argparse

from easydict import EasyDict
import yaml
data_cfg = EasyDict()

import models.models as models
from dataloader.utils import load_poses, load_calib, load_files

class DemoDataset(Dataset):
    def __init__(self, cfg, data_root,split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = data_root

        # Pose information
        self.transform = self.cfg["DATA"]["TRANSFORM"]
        self.poses = {}
        self.filename_poses = cfg["DATA"]["POSES"]

        # Semantic information
        self.semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]

        self.split = split
        if self.split == "train":
            self.training = True 
            self.sequences = self.cfg["DATA"]["SPLIT"]["TRAIN"]
        elif self.split == "val":
            self.training = False
            self.sequences = self.cfg["DATA"]["SPLIT"]["VAL"]
        elif self.split == "test":
            self.training = False
            self.sequences = self.cfg["DATA"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")
        self.point_cloud_range = np.array(self.cfg["DATA"]["POINT_CLOUD_RANGE"],dtype=np.float32)
 

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"

        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            path_to_seq = os.path.join(self.root_dir, seqstr)

            scan_path = os.path.join(path_to_seq, "velodyne")
            self.filenames[seq] = load_files(scan_path)
            if self.transform:
                self.poses[seq] = self.read_poses(path_to_seq)
                assert len(self.poses[seq]) == len(self.filenames[seq])
            else:
                self.poses[seq] = []

            # Get number of sequences based on number of past steps
            n_samples_sequence = max(
                0, len(self.filenames[seq]) - self.skip * (self.n_past_steps - 1)
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.skip * (self.n_past_steps - 1) + sample_idx
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence



    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load point clouds and get sequence

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        seq, scan_idx = self.idx_mapper[idx]

        # set past point clouds path
        from_idx = scan_idx - self.skip * (self.n_past_steps - 1)
        to_idx = scan_idx + 1
        past_indices = list(range(from_idx, to_idx, self.skip))
        past_files = self.filenames[seq][from_idx : to_idx : self.skip]
        
        #load pointcloud
        list_past_point_clouds_raw = [self.read_point_cloud(f) for f in past_files]

        for i,pcd in enumerate(list_past_point_clouds_raw):
            if self.transform:
                from_pose = self.poses[seq][past_indices[i]]
                to_pose = self.poses[seq][past_indices[-1]]
                pcd[:,:3] = self.transform_point_cloud(pcd[:,:3], from_pose, to_pose)
            list_past_point_clouds_raw[i] = pcd

        #pointcloude forward process
        list_past_point_clouds = []
        for i in range(0,len(list_past_point_clouds_raw)):
            list_past_point_clouds.append(torch.from_numpy(list_past_point_clouds_raw[i])) 
            
        for i, pcd in enumerate(list_past_point_clouds):
            time_index = i - self.n_past_steps + 1
            timestamp = round(time_index * self.dt_pred, 3)
            list_past_point_clouds[i]=self.timestamp_tensor(pcd[:,:4], timestamp)

        past_point_clouds = torch.cat(list_past_point_clouds, dim=0) 
        meta = (seq,scan_idx,past_files)

        data_dict = {  
            "meta":meta,
            "past_point_clouds":past_point_clouds,  # x, y, z, r, t
            "batch_size_npast":self.n_past_steps, 
        }     
        return data_dict

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = np.linalg.inv(to_pose) @ from_pose
        NP = past_point_clouds.shape[0]
        xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds

    def read_ground_label(self,filename):
        labels = np.fromfile(filename,dtype=np.uint32)
        labels = labels.reshape(-1)
        labels = labels & 0xFFFF
        return labels

    def timestamp_tensor(self,tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * torch.ones((n_points, 1))
        timestamped_tensor = torch.hstack([tensor, time])
        return timestamped_tensor
    
    def read_poses(self, path_to_seq):
        pose_file = os.path.join(path_to_seq, self.filename_poses)
        calib_file = os.path.join(path_to_seq, "calib.txt")
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
        poses = np.array(new_poses)
        return poses

    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))
        return point_cloud

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        if os.path.isfile(filename):
            labels = np.fromfile(filename, dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF  # Mask semantics in lower half
            mapped_labels = copy.deepcopy(labels)
            for k, v in self.semantic_config["learning_map"].items():
                mapped_labels[labels == k] = v
            selected_labels = torch.Tensor(mapped_labels.astype(np.float32)).long()
            selected_labels = selected_labels.reshape((-1, 1))
            return selected_labels
        else:
            return torch.Tensor(1, 1).long()

    def read_bounding_box_label(self,filename):
        """Load object boundingbox  from .npy file"""
        boundingbox_label_load = np.load(filename,allow_pickle=True)
        if len(boundingbox_label_load)==0: 
            boundingbox_label_load = []
            boundingbox_label_load.append([0,0,1,[0,0,0,0,0,0,0]]) # Prevent list boundingbox_label_load from being empty
        dynamic_falg = False
        boundingbox_label_list = []
        for i in range(0,len(boundingbox_label_load)):
            boundingbox_label = np.zeros(9,dtype=np.float)
            boundingbox_label[0] = boundingbox_label_load[i][1]
            boundingbox_label[1] = boundingbox_label_load[i][2]
            boundingbox_label[2:9] = boundingbox_label_load[i][3][:]
            # Merging label
            if boundingbox_label[0]==1 or boundingbox_label[0] ==3 or boundingbox_label[0]== 6:# (car bus truck) -> car
                boundingbox_label[0]=1
            elif boundingbox_label[0]==8:  # person
                boundingbox_label[0] =2
            elif boundingbox_label[0]==9 or boundingbox_label[0]==10: # (bicyclist motorcyclist) -> cyclist
                boundingbox_label[0] =3
            else:
                boundingbox_label[0] = 0  # other -> 0
            boundingbox_label_list.append(boundingbox_label)
            if boundingbox_label[1] >0:
                dynamic_falg = True

        if dynamic_falg ==False: # Without moving object
            boundingbox_label_list.append([0,1,0,0,0,0,0,0,0]) # add a fake moving object to avoid a empty list

        box_label_numpy = np.array(boundingbox_label_list)
        return box_label_numpy

    @staticmethod
    def collate_batch_test(batch):
        list_data_dict = [item for item in batch]
        return list_data_dict

def parse_config():

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='config/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--split', type=str, default='valid', help='valid or test')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    return args

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()

def to_original_labels(labels, semantic_config):
    original_labels = copy.deepcopy(labels)
    for k, v in semantic_config["learning_map_inv"].items():
        original_labels[labels == k] = v
    return original_labels


def main():
    args = parse_config()
    cfg = torch.load(args.ckpt)["hyper_parameters"]
    data_root = args.data_path
    split = args.split

    cfg["TRAIN"]["BATCH_SIZE"] = 1
    
    # for validation
    if split=='valid':
        sequences = [8]
    elif split=='test':
        sequences = [11,12,13,14,15,16,17,18,19,20,21]

    id = cfg["EXPERIMENT"]["ID"]

    n_past_origin  = cfg["MODEL"]["N_PAST_STEPS"]
    n_delta_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]

    for seq_idx in sequences: 
        cfg["DATA"]["SPLIT"]["TEST"]  = [seq_idx]
        print("-----------Predict the first ", cfg["MODEL"]["N_PAST_STEPS"]," point clouds----------")
        for i in tqdm(range(0,int(cfg["MODEL"]["N_PAST_STEPS"]*cfg["MODEL"]["DELTA_T_PREDICTION"]*10))): # infer the N first scans of the sequences
            cfg["MODEL"]["N_PAST_STEPS"] = i+1
            cfg["MODEL"]["DELTA_T_PREDICTION"] = 0.1
            demo_dataset =  DemoDataset(cfg,data_root,split="test")
            demo_loader = DataLoader(
                    dataset=demo_dataset,
                    batch_size=cfg["TRAIN"]["BATCH_SIZE"],
                    collate_fn=demo_dataset.collate_batch_test,
                    shuffle=False,
                    num_workers=cfg["DATA"]["NUM_WORKER"],
                    pin_memory=True,
                    drop_last=False,
                    timeout=0,
            )
            semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
            ignore_index = [
                key for key, ignore in semantic_config["learning_ignore"].items() if ignore
            ]
            model = models.InsMOSNet.load_from_checkpoint(args.ckpt, hparams=cfg)
            model.cuda()
            model.eval()
            with torch.no_grad():
                    Model_mode = 'test'
                    for batch_idx, data_dict in enumerate(demo_loader):
                        path_list = []
                        ground_label_list = []
                        for data in data_dict:
                            seq,idx,past_indice = data['meta']
                            path_list.append(past_indice[-1])
                            if data.get('past_point_clouds') != None:
                                data["past_point_clouds"] = data["past_point_clouds"].cuda()
                            if data.get('past_labels') != None:
                                for j in range(len(data["past_labels"])):
                                    data["past_labels"][j] = data["past_labels"][j].cuda()
                            if data.get('gt_boxes') != None:
                                data["gt_boxes"] = data["gt_boxes"].cuda()
                            if data.get('ground_labels') != None:
                                ground_label_list.append(data["ground_labels"][-1])
                        path_mos = os.path.join("preb_out",id,"mos_preb","sequences",str(seq).zfill(2),"predictions")
                        path_mos_confidence = os.path.join("preb_out",id,"confidence","sequences",str(seq).zfill(2),"predictions")
                        path_bbox = os.path.join("preb_out",id,"bbox_preb","sequences",str(seq).zfill(2),"predictions")
                
                        os.makedirs(path_mos,exist_ok=True)
                        os.makedirs(path_mos_confidence,exist_ok=True)
                        os.makedirs(path_bbox,exist_ok=True)
                        #前向推理
                        preb_dict_list, recall_dict_list,preb_mos_lable_list  = model.forward(data_dict,Model_mode)
                        for file_idx in range(len(path_list)):
                            file_name_mos = os.path.join(path_mos,str(path_list[file_idx])[-10:-4]+".label")
                            file_name_moving = os.path.join(path_mos_confidence,str(path_list[file_idx])[-10:-4]+".npy")
                            file_name_bbox = os.path.join(path_bbox,str(path_list[file_idx])[-10:-4]+".npy")
                            
                            mos_label = preb_mos_lable_list[file_idx].cpu().numpy()
                            mos_label[:, ignore_index] = -float("inf")

                            mos_label_tensor = torch.from_numpy(mos_label)
                            pred_softmax = F.softmax(mos_label_tensor, dim=1)
                            #=========================================
                            pred_softmax_cpu = pred_softmax.detach().cpu().numpy()
                            moving_confidence = pred_softmax_cpu[:, 1:]
                            moving_confidence_label = moving_confidence
                            np.save(file_name_moving, moving_confidence_label)
                            #=======================================
                            pred_labels = torch.argmax(pred_softmax, axis=1).long() #
                            preb_mos_label = pred_labels.cpu().numpy()
                            preb_mos_label = to_original_labels(preb_mos_label, semantic_config)
                            preb_mos_label = preb_mos_label.reshape((-1)).astype(np.int32)

                            preb_dict = preb_dict_list[file_idx][0]
                            for key in preb_dict:
                                preb_dict[key] = preb_dict[key].cpu().numpy()

                            preb_mos_label.tofile(file_name_mos)
                            np.save(file_name_bbox,preb_dict)
                        torch.cuda.empty_cache()
                        break

        #except the N first scans
        cfg["MODEL"]["N_PAST_STEPS"] = n_past_origin
        cfg["MODEL"]["DELTA_T_PREDICTION"] = n_delta_prediction
        demo_dataset =  DemoDataset(cfg,data_root,split="test")
        demo_loader = DataLoader(
                dataset=demo_dataset,
                batch_size=cfg["TRAIN"]["BATCH_SIZE"],
                collate_fn=demo_dataset.collate_batch_test,
                shuffle=False,
                num_workers=cfg["DATA"]["NUM_WORKER"],
                pin_memory=True,
                drop_last=False,
                timeout=0,
        )
        #logger.info(f'Total number of samples: \t{len(demo_dataset)}')

        semantic_config = yaml.safe_load(open(cfg["DATA"]["SEMANTIC_CONFIG_FILE"]))
        ignore_index = [
            key for key, ignore in semantic_config["learning_ignore"].items() if ignore
        ]
        model = models.InsMOSNet.load_from_checkpoint(args.ckpt, hparams=cfg)
        model.cuda()
        model.eval()
        with torch.no_grad():
            Model_mode = 'test'
            for batch_idx, data_dict in enumerate(tqdm(demo_loader)):
                path_list = []
                ground_label_list = []
                for data in data_dict:
                    seq,idx,past_indice = data['meta']
                    path_list.append(past_indice[-1])
                    if data.get('past_point_clouds') != None:
                        data["past_point_clouds"] = data["past_point_clouds"].cuda()
                    if data.get('past_labels') != None:
                        for j in range(len(data["past_labels"])):
                            data["past_labels"][j] = data["past_labels"][j].cuda()
                    if data.get('gt_boxes') != None:
                        data["gt_boxes"] = data["gt_boxes"].cuda()
                    if data.get('ground_labels') != None:
                        ground_label_list.append(data["ground_labels"][-1])
                path_mos = os.path.join("preb_out",id,"mos_preb","sequences",str(seq).zfill(2),"predictions")
                path_mos_confidence = os.path.join("preb_out",id,"confidence","sequences",str(seq).zfill(2),"predictions")
                path_bbox = os.path.join("preb_out",id,"bbox_preb","sequences",str(seq).zfill(2),"predictions")
        
                os.makedirs(path_mos,exist_ok=True)
                os.makedirs(path_mos_confidence,exist_ok=True)
                os.makedirs(path_bbox,exist_ok=True)

                # inference
                preb_dict_list, recall_dict_list,preb_mos_lable_list  = model.forward(data_dict,Model_mode)
                for file_idx in range(len(path_list)):
                    file_name_mos = os.path.join(path_mos,str(path_list[file_idx])[-10:-4]+".label")
                    file_name_moving = os.path.join(path_mos_confidence,str(path_list[file_idx])[-10:-4]+".npy")
                    file_name_bbox = os.path.join(path_bbox,str(path_list[file_idx])[-10:-4]+".npy")
                    
                    mos_label = preb_mos_lable_list[file_idx].cpu().numpy()
                    mos_label[:, ignore_index] = -float("inf")

                    mos_label_tensor = torch.from_numpy(mos_label)
                    pred_softmax = F.softmax(mos_label_tensor, dim=1)
                    #=========================================
                    pred_softmax_cpu = pred_softmax.detach().cpu().numpy()
                    moving_confidence = pred_softmax_cpu[:, 1:]
                    moving_confidence_label = moving_confidence
                    np.save(file_name_moving, moving_confidence_label)
                    #=======================================
                    pred_labels = torch.argmax(pred_softmax, axis=1).long() 
                    preb_mos_label = pred_labels.cpu().numpy()
                    preb_mos_label = to_original_labels(preb_mos_label, semantic_config)
                    preb_mos_label = preb_mos_label.reshape((-1)).astype(np.int32)

                    preb_dict = preb_dict_list[file_idx][0]
                    for key in preb_dict:
                        preb_dict[key] = preb_dict[key].cpu().numpy()

                    preb_mos_label.tofile(file_name_mos)
                    np.save(file_name_bbox,preb_dict)

                torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()