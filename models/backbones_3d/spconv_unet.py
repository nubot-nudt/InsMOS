#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from functools import partial

import torch
import torch.nn as nn
import time
import os
from typing import Set
import numpy as np

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

from spconv.pytorch.utils import gather_features_by_pc_voxel_id

from ..backbones_2d.height_compression import HeightCompression
from ..backbones_2d.base_bev_backbone import BaseBEVBackbone
from ..backbones_2d.center_head import CenterHead

from ..post_process import post_processing
from ..utils import Array_Index


def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class UNetV2(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,mos_class, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None
        
        #=============================Instance Detection=============================
        self.post_process = model_cfg["MODEL"]["POST_PROCESSING"]
        self.num_class = model_cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"]
        self.point_cloud_range = np.array(model_cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        self.voxel_size = model_cfg["DATA"]['VOXEL_SIZE']  #
        self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size) 
        self.grid_size=np.round(self.grid_size).astype(np.int64) 
        self.to_bev = HeightCompression(model_cfg["MODEL"]["MAP_TO_BEV"])  # to bev
        self.bev_backbone = BaseBEVBackbone(model_cfg["MODEL"]["BACKBONE_2D"],input_channels=self.to_bev.num_bev_features) # 2d cnn
        self.center_head = CenterHead(model_cfg["MODEL"]["DENSE_HEAD"],
                                        input_channels=model_cfg["MODEL"]["BACKBONE_2D"]["NUM_UPSAMPLE_FILTERS"][0],
                                        num_class=model_cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"] if not model_cfg["MODEL"]["DENSE_HEAD"]["CLASS_AGNOSTIC"] else 1,
                                        class_names=model_cfg["MODEL"]["DENSE_HEAD"]["CLASE_NAME"],
                                        grid_size = self.grid_size,
                                        point_cloud_range=self.point_cloud_range,
                                        predict_boxes_when_training=model_cfg.get('ROI_HEAD', False))


        #=============================Upsample Fusion===================================
        self.inv_conv_out = spconv.SparseInverseConv3d(128,128,(3,1,1),bias=False,indice_key='spconv_down2')
        self.conv_up_instance_block= block(128+self.num_class, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.conv_up_instance_block_up4 = block(64+self.num_class, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.conv_up_instance_block_up3 = block(32+self.num_class, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.conv_up_instance_block_up2 = block(16+self.num_class, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv_up_instance_block_up1 = block(16+self.num_class, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')

        # decoder
        self.conv_up_t4 = SparseBasicBlock(128, 128, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')


        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv_up_out = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.mos_seg_layer = nn.Linear(16, mos_class, bias=True)
        self.num_point_features = 16



    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    
    def forward(self, batch_dict,Model_mode):
        """
        Args:
            Model_mode: train || eval || test
            batch_dict:
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx] , set batch_idx=1
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        
        # -----------------batch_dict['voxel_features'] include original features and motion features of LiDAR points------------------------
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size=  1 # the batch_size just for voxel processing

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        #-----------------generate spatio-temporal features by fusing original features{x,y,z,r} with motion features----------------------
        # Why do we call it a spatio-temporal feature? 
        # Because "batch_dict['voxel_features']={x,y,z,r,motion_feature}" contain motion features and spatial features inside, and motion features can be understood as temporal features, 
        # but motionnet not only extracts temporal features, it also extracts certain spatial features, so we call the output of motionnet as motion features. 
        # And here the features mainly extracted from inside "batch_dict['voxel_features']={x,y,z,r,motion_feature}" are partly used for instance detection and partly used for up-sampling fusion, 
        # this feature contains both temporal and spatial features, so we call it spatio-temporal feature.
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # the spatio-temporal features partly used for instance detection
        #========================instance detection===========================
        batch_dict = self.to_bev(batch_dict)  # BEV spaces
        batch_dict["current_bev"] = batch_dict["spatial_features"][-1].unsqueeze(0)
        batch_dict = self.bev_backbone(batch_dict)
        batch_dict = self.center_head(batch_dict,Model_mode)
        # --------------nms... for refine bounding box ----------------
        pred_dicts, recall_dicts = post_processing(batch_dict,self.post_process,self.num_class)


        # the spatio-temporal features partly used for  up-sampling fusion
        sparse_inv_bev = self.inv_conv_out(batch_dict["encoded_spconv_tensor"])
        #========================upsample fusion===========================
        # --------------------instance layer 1 ----------------------
        boxes = pred_dicts[0]["pred_boxes"].clone().detach()
        boxes_label = pred_dicts[0]["pred_labels"].clone().detach()
        boxes_label = boxes_label.view(-1,1)
        boxes[:,0] = (boxes[:,0] -self.point_cloud_range[0])/self.voxel_size[0]/batch_dict['encoded_spconv_tensor_stride']
        boxes[:,1] = (boxes[:,1] -self.point_cloud_range[1])/self.voxel_size[1]/batch_dict['encoded_spconv_tensor_stride']
        boxes[:,2] = (boxes[:,2] -self.point_cloud_range[2])/self.voxel_size[2]/batch_dict['encoded_spconv_tensor_stride']
        boxes[:,3] = boxes[:,3]/self.voxel_size[0]/batch_dict['encoded_spconv_tensor_stride']
        boxes[:,4] = boxes[:,4]/self.voxel_size[1]/batch_dict['encoded_spconv_tensor_stride']
        boxes[:,5] = boxes[:,5]/self.voxel_size[2]/batch_dict['encoded_spconv_tensor_stride']
        boxes = torch.hstack([boxes,boxes_label])
        sparse_inv_bev_features =np.zeros((sparse_inv_bev.features.shape[0],self.num_class),dtype=int)
        sparse_inv_bev_coord = sparse_inv_bev.indices.detach().cpu().numpy()[:,[0,3,2,1]]
        sparse_inv_bev_coord = sparse_inv_bev_coord[:,1:4]
        
        # find and extract points in instance bounding box to generate instance features, which is mainly for point-level fusion
        # this function is achieved by pybind for quickly search points
        # find_features_by_bbox_with_yaw:
            # input: scan: LiDAR points (N x 4)
            #        features: zeros(N x 3)
            # output: (N x 3), 0 colums is car, 1 colums is pedestrian, 2 colums is cyclist, The value stored is the 0 or 1
            # e.g. [ 1, 0 ,0] represent the point is car 
            #      [ 0, 1 ,0] represent the point is pedestrian 
            #      [ 0, 0 ,1] represent the point is cyclist
        features_instance = Array_Index.find_features_by_bbox_with_yaw(sparse_inv_bev_coord,boxes.detach().cpu().numpy(),sparse_inv_bev_features)
        features_instance_tensor = torch.from_numpy(features_instance)
        features_instance_tensor = features_instance_tensor.to(device=sparse_inv_bev.features.device)
        
        # ----------------------concatenate instance features and spatio-temporal features -------------------------------------
        sparse_inv_bev_instance = replace_feature(sparse_inv_bev, torch.cat([sparse_inv_bev.features,features_instance_tensor],dim=1)) 

        # ----------------------fuse instance features and spatio-temporal features by convolution-----------
        x_conv_instance=self.conv_up_instance_block(sparse_inv_bev_instance) 
        x_up4 = self.UR_block_forward(x_conv_instance, x_conv_instance, self.conv_up_t4, self.conv_up_m4, self.inv_conv4) 


        # --------------------instance layer 2 ----------------------
        boxes[:,0:6] = 2*boxes[:,0:6]
        x_up4_instance_feature = np.zeros((x_up4.features.shape[0],self.num_class),dtype=int)
        x_up4_coord = x_up4.indices.detach().cpu().numpy()[:,[0,3,2,1]]
        x_up4_coord = x_up4_coord[:,1:4]
        x_up4_fea_inst = Array_Index.find_features_by_bbox_with_yaw(x_up4_coord,boxes.detach().cpu().numpy(),x_up4_instance_feature)
        x_up4_fea_inst_tensor = torch.from_numpy(x_up4_fea_inst)
        x_up4_fea_inst_tensor = x_up4_fea_inst_tensor.to(device=x_up4.features.device)

        # ---------------------fusion-------------------------
        x_up4_instance = replace_feature(x_up4, torch.cat([x_up4.features,x_up4_fea_inst_tensor],dim=1))
        x_up4_conv_inst=self.conv_up_instance_block_up4(x_up4_instance)  
        x_up3 = self.UR_block_forward(x_conv3, x_up4_conv_inst, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)


        # --------------------instance layer 3 ----------------------
        boxes[:,0:6] = 2*boxes[:,0:6]
        x_up3_instance_feature = np.zeros((x_up3.features.shape[0],self.num_class),dtype=int)
        x_up3_coord = x_up3.indices.detach().cpu().numpy()[:,[0,3,2,1]]
        x_up3_coord = x_up3_coord[:,1:4]
        x_up3_fea_inst = Array_Index.find_features_by_bbox_with_yaw(x_up3_coord,boxes.detach().cpu().numpy(),x_up3_instance_feature)
        x_up3_fea_inst_tensor = torch.from_numpy(x_up3_fea_inst)
        x_up3_fea_inst_tensor = x_up3_fea_inst_tensor.to(device=x_up3.features.device)

        # ---------------------fusion-------------------------
        x_up3_instance = replace_feature(x_up3, torch.cat([x_up3.features,x_up3_fea_inst_tensor],dim=1))
        x_up3_conv_inst=self.conv_up_instance_block_up3(x_up3_instance) #融合了instance信息35->32
        x_up2 = self.UR_block_forward(x_conv2, x_up3_conv_inst, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)


        # --------------------instance layer 4 ----------------------
        boxes[:,0:6] = 2*boxes[:,0:6]
        x_up2_instance_feature = np.zeros((x_up2.features.shape[0],self.num_class),dtype=int)
        x_up2_coord = x_up2.indices.detach().cpu().numpy()[:,[0,3,2,1]]
        x_up2_coord = x_up2_coord[:,1:4]
        x_up2_fea_inst = Array_Index.find_features_by_bbox_with_yaw(x_up2_coord,boxes.detach().cpu().numpy(),x_up2_instance_feature)
        x_up2_fea_inst_tensor = torch.from_numpy(x_up2_fea_inst)
        x_up2_fea_inst_tensor = x_up2_fea_inst_tensor.to(device=x_up2.features.device)

        # ---------------------fusion-------------------------
        x_up2_instance = replace_feature(x_up2, torch.cat([x_up2.features,x_up2_fea_inst_tensor],dim=1))
        x_up2_conv_inst=self.conv_up_instance_block_up2(x_up2_instance)  
        x_up1 = self.UR_block_forward(x_conv1, x_up2_conv_inst, self.conv_up_t1, self.conv_up_m1, self.conv_up_out)

        x_up1_instance = replace_feature(x_up1, torch.cat([x_up1.features,x_up2_fea_inst_tensor],dim=1))
        x_up1_conv_inst=self.conv_up_instance_block_up1(x_up1_instance) #
        
        
        current_seg_feature = x_up1_conv_inst.features
        mos_seg_preb = self.mos_seg_layer(current_seg_feature)
        current_pc_voxel_id = batch_dict["list_pc_voxel_id"][-1]

        # ------------features: from voxels to points , generate point-level motion feature----------------
        point_seg_feature = gather_features_by_pc_voxel_id(mos_seg_preb,current_pc_voxel_id)


        if Model_mode =='train':
            return self.center_head.get_loss(),point_seg_feature
        else:
            return point_seg_feature,pred_dicts, recall_dicts


