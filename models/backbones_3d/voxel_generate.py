#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel

class VoxelGenerate(nn.Module):
    def __init__(self,voxel_size,point_cloud_range,max_number_of_voxel,max_point_per_voxel,num_point_feature):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range=point_cloud_range
        self.max_number_of_voxel=max_number_of_voxel
        self.max_point_per_voxel=max_point_per_voxel
        self.num_point_feature=num_point_feature
    
    def forward(self,batch_dict):
        #batch_dict["current_point"] = batch_dict["past_point_clouds"][batch_dict["past_point_clouds"][:,-1]==0,:4]
        self.voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz = self.point_cloud_range,
                num_point_features=self.num_point_feature, # feature dim
                max_num_voxels = self.max_number_of_voxel,
                max_num_points_per_voxel = self.max_point_per_voxel,
                device=batch_dict["current_point"].device
            )
        voxel_output=self.voxel_generator.generate_voxel_with_id(batch_dict["current_point"])
        batch_dict['voxels'],batch_dict['voxel_coords'],batch_dict['voxel_num_points'],batch_dict['pc_voxel_id']=voxel_output
        batch_dict['voxel_coords'] = torch.hstack([torch.zeros((batch_dict['voxel_coords'].shape[0],1),device=batch_dict['voxel_coords'].device),batch_dict['voxel_coords']])
        batch_dict["list_pc_voxel_id"] = [batch_dict['pc_voxel_id']]
        return batch_dict


