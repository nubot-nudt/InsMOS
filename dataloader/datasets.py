#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


import numpy as np
import yaml
import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .utils import load_poses, load_calib, load_files
from .augmentation import (
    shift_point_cloud,
    rotate_point_cloud,
    jitter_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
    random_rotation,
    random_scaling,
    random_flip
)

from .data_processor import DataProcessor
from collections import defaultdict

class KittiSequentialDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("DATA")

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
        self.data_processor = DataProcessor(
            self.cfg["DATA_PROCESSOR"], point_cloud_range=self.point_cloud_range, training=self.training
        )    

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
        
        #load bounding box 
        bounding_box_file = [ os.path.join(self.root_dir, str(seq).zfill(2), "boundingbox_label", str(i).zfill(6) + ".npy")
            for i in past_indices ]

        list_bounding_box = [self.read_bounding_box_label(bounding_box_file[-1])] # only need the current bounding box label
        gt_box = np.zeros([len(list_bounding_box[0]),8])
        for i, boxs in enumerate(list_bounding_box):
            gt_box[:len(boxs),0:7] = boxs[:,2:9]
            gt_box[:len(boxs),7] = boxs[:,0]

        list_past_point_clouds_raw = [self.read_point_cloud(f) for f in past_files]
        num_point_list=[]
        for i,pcd in enumerate(list_past_point_clouds_raw):
            if self.transform:
                from_pose = self.poses[seq][past_indices[i]]
                to_pose = self.poses[seq][past_indices[-1]]
                pcd[:,:3] = self.transform_point_cloud(pcd[:,:3], from_pose, to_pose)
                num_point_list.append(pcd.shape[0])
            list_past_point_clouds_raw[i] = pcd

        if self.training:
            gt_box_for_augment = gt_box[:,0:7]
            past_points_for_transform=np.concatenate(list_past_point_clouds_raw,axis=0)
            past_points_for_transform,gt_box_for_augment=random_flip(past_points_for_transform,gt_box_for_augment,['x'])
            past_points_for_transform,gt_box_for_augment=random_rotation(past_points_for_transform,gt_box_for_augment,[-0.78539816, 0.78539816])
            past_points_for_transform,gt_box_for_augment=random_scaling(past_points_for_transform,gt_box_for_augment,[0.95, 1.05])

            for i in range(len(num_point_list)):
                if i==0:
                    list_past_point_clouds_raw[i]=past_points_for_transform[0:num_point_list[0],:]
                else:
                    list_past_point_clouds_raw[i]=past_points_for_transform[np.sum(num_point_list[0:i]):np.sum(num_point_list[0:i+1]),:]
            gt_box[:,0:7] = gt_box_for_augment

        # Load past labels
        label_files = [
            os.path.join(self.root_dir, str(seq).zfill(2), "labels", str(i).zfill(6) + ".label")
            for i in past_indices
        ]
        past_labels = [self.read_labels(f) for f in label_files]

        list_past_point_clouds = []
        for i in range(0,len(list_past_point_clouds_raw)):
            data_point={'points':np.hstack([list_past_point_clouds_raw[i],past_labels[i]])}
            data_point_output = self.data_processor.forward(data_dict=data_point)
            list_past_point_clouds.append(torch.tensor(data_point_output["points"],dtype=torch.float32)) 
            past_labels[i] = torch.tensor(data_point_output["points"][:,-1],dtype=torch.float32)

        for i, pcd in enumerate(list_past_point_clouds):
            time_index = i - self.n_past_steps + 1
            timestamp = round(time_index * self.dt_pred, 3)
            list_past_point_clouds[i] = self.timestamp_tensor(pcd[:,:4], timestamp)

        past_point_clouds = torch.cat(list_past_point_clouds, dim=0) 

        gt_boxes = torch.tensor(gt_box.astype(np.float32)).unsqueeze(0)
        meta = (seq, scan_idx, past_indices)
        data_dict = {
            "meta":meta,   
            "past_point_clouds":past_point_clouds,  
            "past_labels":past_labels,               
            "gt_boxes":gt_boxes,
            "batch_size_npast":self.n_past_steps, 
        }

        return data_dict

    def transform_point_cloud(self, past_point_clouds, from_pose, to_pose):
        transformation = np.linalg.inv(to_pose) @ from_pose
        NP = past_point_clouds.shape[0]
        xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds


    def read_point_cloud(self, filename):
        """Load point clouds from .bin file"""
        point_cloud = np.fromfile(filename, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 4))
        #point_cloud = point_cloud[:, :3]
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
            boundingbox_label_load.append([0,0,1,[0,0,0,0,0,0,0]]) 
        dynamic_falg = False
        boundingbox_label_list = []
        for i in range(0,len(boundingbox_label_load)):
            boundingbox_label = np.zeros(9,dtype=np.float)
            boundingbox_label[0] = boundingbox_label_load[i][1]
            boundingbox_label[1] = boundingbox_label_load[i][2]
            boundingbox_label[2:9] = boundingbox_label_load[i][3][:]
            # merage label
            if boundingbox_label[0]==1 or boundingbox_label[0] ==3 or boundingbox_label[0]== 6:
                boundingbox_label[0]=1
            elif boundingbox_label[0]==8:  
                boundingbox_label[0] =2
            elif boundingbox_label[0]==9 or boundingbox_label[0]==10: 
                boundingbox_label[0] =3
            else:
                boundingbox_label[0] = 0  
            boundingbox_label_list.append(boundingbox_label)
            if boundingbox_label[1] >0:
                dynamic_falg = True

        if dynamic_falg ==False: 
            boundingbox_label_list.append([0,1,0,0,0,0,0,0,0])

        box_label_numpy = np.array(boundingbox_label_list)
        return box_label_numpy


    @staticmethod
    def timestamp_tensor(tensor, time):
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
