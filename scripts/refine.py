#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import json
import argparse
import os
import sys
sys.path.append(".")
import yaml
import open3d as o3d
import copy
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel
import time
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Array_Index

import numpy.linalg as LA

def load_labels(label_path):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    #print("label shape:",label.shape)

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    assert ((sem_label + (inst_label << 16) == label).all())
    return sem_label, inst_label

def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    Args:
      pose_path: (Complete) filename for the pose file
    Returns:
      A numpy array of size nx4x4 with n poses as 4x4 transformation
      matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")

                    if len(T_w_cam0) == 12:
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    elif len(T_w_cam0) == 16:
                        T_w_cam0 = T_w_cam0.reshape(4, 4)
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]

    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")
    return np.array(poses)

def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file."""

    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print("Calibrations are not avaialble.")
    return np.array(T_cam_velo)

def get_lidar_pose(posefile,calibfile):
    pose_in_cam = load_poses(posefile)
    
    inv_frame0 = np.linalg.inv(pose_in_cam[0])
    # load calibrations
    T_cam_velo = load_calib(calibfile)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in pose_in_cam:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo)) 
    pose_in_lidar = np.array(new_poses)
    return pose_in_lidar

def load_confidence(mos_label_path):
    mos_label = np.load(mos_label_path)
    mos_label = mos_label.reshape((-1,3))
    mos_label_tensor = torch.from_numpy(mos_label)
    pred_softmax = F.softmax(mos_label_tensor, dim=1)
    pred_softmax = pred_softmax.cpu().numpy()
    moving_confidence = pred_softmax[:, 2]
    return moving_confidence

def load_scan(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32).reshape((-1,4))
    return scan

def load_mos_label(mos_label_path):
    ignore_index = [0]
    mos_label = np.load(mos_label_path)

    mos_label = mos_label.reshape((-1,3))
    mos_label[:, ignore_index] = -float("inf")
    mos_label_tensor = torch.from_numpy(mos_label)
    pred_softmax = F.softmax(mos_label_tensor, dim=1)
    pred_labels = torch.argmax(pred_softmax, axis=1).long() # 
    return pred_labels
    
def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    transformation = np.linalg.inv(to_pose) @ from_pose
    NP = past_point_clouds.shape[0]
    xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
    past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds

def to_original_labels(labels, semantic_config):
    original_labels = copy.deepcopy(labels)
    for k, v in semantic_config["learning_map_inv"].items():
        original_labels[labels == k] = v
    return original_labels

def main(data_path,split):
    semantic_mask_config = yaml.safe_load(open("./config/semantic-kitti-mos.yaml"))

    if split=='valid':
        sequences = [8]
    elif split=='test':
        sequences = [11,12,13,14,15,16,17,18,19,20,21]

    for seqs in sequences:
        seqs = str(seqs).zfill(2)
        data_dir = os.path.join(data_path, seqs)
        scan_folder = os.path.join(data_dir, 'velodyne')
        preb_bbox_label = os.path.join("./preb_out/InsMOS/bbox_preb",'sequences', seqs, 'predictions')
        mos_label_folder = os.path.join("./preb_out/InsMOS/mos_preb",'sequences', seqs, 'predictions')
        moving_label_folder = os.path.join("./preb_out/InsMOS/confidence",'sequences',seqs,'predictions')

        path_mos = os.path.join("preb_out_refine","mos_preb","sequences",str(seqs).zfill(2),"predictions")
        os.makedirs(path_mos,exist_ok=True)

        scan_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_folder)) for f in fn]
        scan_paths.sort()
        preb_bbox_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(preb_bbox_label)) for f in fn]
        preb_bbox_path.sort()
        mos_label_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(mos_label_folder)) for f in fn]
        mos_label_paths.sort()
        moving_label_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(moving_label_folder)) for f in fn]
        moving_label_path.sort()

        posefile = os.path.join(data_dir, 'poses.txt')
        calibfile = os.path.join(data_dir, 'calib.txt')
        lidar_pose = get_lidar_pose(posefile,calibfile) 

        instance_moving_attribute_widow = []
        instance_window = 5
        for frame_idx in tqdm(range(0,len(scan_paths))):
            scan = load_scan(scan_paths[frame_idx])
            preb_bbox = np.load(preb_bbox_path[frame_idx],allow_pickle=True)
            preb_bbox = preb_bbox.item()
    
            mos_label,_ = load_labels(mos_label_paths[frame_idx])
            moving_confidence = np.load(moving_label_path[frame_idx])
            moving_confidence = moving_confidence.reshape(-1,2)
            if frame_idx<9:
                moving_confidence = np.zeros([mos_label.shape[0],2])

            mos_label=mos_label.astype(np.int32)
            mos_label[mos_label==251]=2
            mos_label[mos_label==9]=1

            boxes = np.concatenate((preb_bbox['pred_boxes'],preb_bbox['pred_labels'].reshape(-1,1)),axis=1)
            features =np.zeros((scan.shape[0],3),dtype=int)

            # find and search points in instance bounding box
            # find_point_in_instance_bbox_with_yaw:
            # input: scan: LiDAR points (N x 4)
            #        features: zeros(N x 3)
            #        ground_height: float (remove ground in the bounding box)
            # output: (N x 3), 0 colums is car, 1 colums is pedestrian, 2 colums is cyclist, The value stored is the id of the instance
            # e.g. [ 1, 0 ,0] represent the point is car and the id is 1
            #      [ 0, 8 ,0] represent the point is pedestrian and the id is 8
            #      [ 0, 0 ,12] represent the point is cyclist and the id is 12
            index = Array_Index.find_point_in_instance_bbox_with_yaw(scan,boxes,features,0.03) # 0.03


            # colors = np.ones((scan.shape[0],3),dtype=np.float)
            # colors[sum_index==0] = [0.5,0.5,0.5]
            moving_car_num = 0
            instance_car_idx_list = []
            instance_car_idx_moving_list = []
            instance_car_all_list = []
            instance_moving_attribute_list = []
            car_idx = -1

            #  Since the prediction of pedestrians and cyclists through InsMOS has yielded good results, we will only consider cars here.
            # ==============================bottom up============================================
            for instance_idx in range(0,len(preb_bbox['pred_labels'])):
                if preb_bbox['pred_labels'][instance_idx] ==1: # just consider cars
                    # colors[index[:,0]==(indtsnce_idx+1)] = [indtsnce_idx*0.1,0,indtsnce_idx*0.05]

                    instance_car_idx = np.where(index[:,0]==instance_idx+1)[0]
                    instance_car_point = len(instance_car_idx)
                    if instance_car_point!=0:
                        instance_car_mos = mos_label[instance_car_idx]
                        instance_car_moving_point = len(np.where(instance_car_mos==2)[0])
                        car_moving_confidence = moving_confidence[instance_car_idx][:,1]
                        car_moving_confidence_point = len(np.where(car_moving_confidence>=0.00001)[0])

                        # if moving percentage > 0.6 the car is moving
                        car_idx = car_idx+1
                        instance_car_all_list.append(instance_car_idx)
                        moving_attribute = preb_bbox['pred_boxes'][instance_idx]
                        if(instance_car_moving_point/instance_car_point)>0.6:
                            moving_attribute[-1] = 1
                        else:
                            moving_attribute[-1] = 0
                        instance_moving_attribute_list.append(moving_attribute)
            
                        # in low dynamic environment, The percentage of static cars is typically 0 by insmos predicted
                        if(instance_car_moving_point/instance_car_point)>0.3:
                            moving_car_num = moving_car_num +1

                        if (instance_car_moving_point/instance_car_point)>0.001:
                            instance_car_idx_list.append(car_idx)

                        if (car_moving_confidence_point/car_moving_confidence.shape[0])>0.5:
                            instance_car_idx_moving_list.append(car_idx)

            if frame_idx!=0:
                # high dynamic scene
                if moving_car_num>=3: 
                    for instance_car_index in instance_car_idx_list:
                        if frame_idx<instance_window:
                            mos_label[instance_car_all_list[instance_car_index]] = 2
                        instance_moving_attribute_list[instance_car_index][-1] = 1     
                if moving_car_num>=5:
                    for instance_car_moving_index in instance_car_idx_moving_list:
                        if frame_idx<instance_window:
                            mos_label[instance_car_all_list[instance_car_moving_index]] = 2
                        instance_moving_attribute_list[instance_car_moving_index][-1] = 1
            else:   
                if moving_car_num>=5:
                    for instance_car_index in instance_car_idx_list:
                        mos_label[instance_car_all_list[instance_car_index]] = 2
                    for instance_car_moving_index in instance_car_idx_moving_list:
                        mos_label[instance_car_all_list[instance_car_moving_index]] = 2   

            # track instance 
            instance_moving_attribute_widow.append(instance_moving_attribute_list)
            if frame_idx>= instance_window: 
                assert len(instance_moving_attribute_widow)== (instance_window+1)
                instance_attribute_current = instance_moving_attribute_widow[-1].copy()
                for instance_attribute in instance_attribute_current:
                    find_flag = 0
                    moving_flag = 0
                    for i in range(0,instance_window):
                        # transform instance center to past coordinate --> alignment 
                        center_transform = transform_point_cloud(instance_attribute[0:3].reshape(-1,3),lidar_pose[frame_idx],lidar_pose[frame_idx-i-1])
                        center_transform = center_transform.reshape(-1)
                        for instance_attribute_pre in instance_moving_attribute_widow[instance_window-1-i]:
                            # bouning box match , mainly for bouning box dimension
                            if(abs(center_transform[0]-instance_attribute_pre[0])<1 and abs(center_transform[1]-instance_attribute_pre[1])<1 and abs(center_transform[2]-instance_attribute_pre[2])<0.5 
                                and abs(instance_attribute[3]-instance_attribute_pre[3])<0.3 and abs(instance_attribute[4]-instance_attribute_pre[4])<0.3 and abs(instance_attribute[5]-instance_attribute_pre[5])<0.3):
                                find_flag = find_flag + 1
                                if instance_attribute_pre[-1] == 1:
                                    moving_flag = moving_flag + 1
                                break
                    if find_flag == 5:
                        if moving_flag >3 :
                            instance_attribute[-1] = 1
                    else:
                        if (moving_flag >1) or (moving_flag>0 and moving_car_num>=3) :
                            instance_attribute[-1] = 1
                            
                # ===============================top-down==========================================
                for j in range(0,len(instance_attribute_current)):
                    if instance_attribute_current[j][-1] == 1: 
                        mos_label[instance_car_all_list[j]] = 2
                    # If too many cars are present in the scene, there may be some false negative, when(n>6) not going to refine the network predictions,
                    if instance_attribute_current[j][-1] == 0 and len(instance_attribute_current) > 6: 
                        mos_label[instance_car_all_list[j]] = 1

                instance_moving_attribute_widow.pop(0)

            file_name_mos = os.path.join(path_mos,str(mos_label_paths[frame_idx])[-12:-6]+".label")
            preb_mos_label = to_original_labels(mos_label, semantic_mask_config)
            preb_mos_label = preb_mos_label.reshape((-1)).astype(np.int32)

            preb_mos_label.tofile(file_name_mos)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--split', type=str, default='valid',
                        help='valid or test')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    args = parser.parse_args()

    main(args.data_path,args.split)