#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn

import MinkowskiEngine as ME
from models.MinkowskiEngine.customminkunet import CustomMinkUNet

# MotionNet is modified based on 4DMOS.  In fact,
# motionnet can be replaced in other ways, such as residual range image or bev image
class MotionNet(nn.Module):
    def __init__(self,dt_prediction,voxel_size,out_channels):
        super().__init__()
        self.dt_prediction = dt_prediction
        ds = voxel_size[0]
        self.quantization = torch.Tensor([ds, ds, ds, self.dt_prediction])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=out_channels, D=4)

    
    def forward(self,batch_dict):
        past_point_clouds = [torch.hstack([batch_dict["past_point_clouds"][:,0:3], batch_dict["past_point_clouds"][:,4].view(-1,1)])]

        # input :4D Voxel
        quantization = self.quantization.type_as(past_point_clouds[0]) # past_point_clouds:[batch,current+num_past,[x,y,z,t]]
        past_point_clouds = [
            torch.div(point_cloud, quantization) for point_cloud in past_point_clouds
        ]
        features = [
            0.5 * torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]
        coords, features = ME.utils.sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))

        sparse_tensor = tensor_field.sparse()
        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)
        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)

        # current point
        batch_dict["current_point"] = batch_dict["past_point_clouds"][past_point_clouds[0][:,3]==0,:4]

        # get motion feature
        batch_dict["current_motion_feature"] = out.features[past_point_clouds[0][:,3]==0,:]

        # concatenate current point and motion feature
        batch_dict["current_point"] = torch.hstack([batch_dict["current_point"],batch_dict["current_motion_feature"] ]) 

        return batch_dict

