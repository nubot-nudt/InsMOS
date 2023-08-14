#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


import torch
import numpy as np



def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False



def random_rotation(pointcloud_with_batch,gt_box,rot):
  """Input:Nx5 (x,y,z,r,batch)"""
  pointcloud = pointcloud_with_batch[:,0:3]
  rot_range = rot
  if not isinstance(rot_range,list):
    rot_range = [-rot_range,rot_range]

  gt_boxes, points = global_rotation(gt_box, pointcloud, rot_range=rot_range)
  pointcloud_with_batch[:,0:3] = points

  return pointcloud_with_batch,gt_boxes

def random_scaling(pointcloud_with_batch,gt_box,scal_range):
  pointcloud = pointcloud_with_batch[:,0:3]
  gt_boxes,points = global_scaling(gt_box,pointcloud,scal_range)
  pointcloud_with_batch[:,0:3] = points

  return pointcloud_with_batch,gt_boxes

def random_flip(pointcloud_with_batch,gt_box,along_axis_list):
  pointcloud = pointcloud_with_batch[:,0:3]
  for cur_axis in along_axis_list:
    assert cur_axis in ['x','y']
    if cur_axis == 'x':
        gt_boxes,points = random_flip_along_x(gt_box,pointcloud)
    elif cur_axis == 'y':
        gt_boxes,points = random_flip_along_y(gt_box,pointcloud)

    
  
  pointcloud_with_batch[:,0:3] = points
  return pointcloud_with_batch,gt_boxes

def random_shift(pointcloud_with_batch,gt_box,shift_range=0.1):
    pointcloud = pointcloud_with_batch[:,0:3]
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    pointcloud[:, :3] += shifts
    gt_box[:,:3] += shifts
    pointcloud_with_batch[:,0:3] = pointcloud
    return pointcloud_with_batch,gt_box
    
def random_jitter(pointcloud_with_batch,gt_box,sigma=0.01, clip=0.05):
    pointcloud = pointcloud_with_batch[:,0:3]
    assert clip > 0
    N, _ = pointcloud.shape
    jitter = torch.clip(sigma * torch.randn(N, 3), -1 * clip, clip)
    pointcloud[:, :3] += jitter
    gt_box[:,:3] += jitter
    pointcloud_with_batch[:,0:3] = pointcloud
    return pointcloud_with_batch,gt_box

def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points

def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points

def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] =rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot



def rotate_point_cloud(point_cloud):
    """Randomly rotate the point clouds to augument the dataset
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * torch.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = torch.Tensor([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    point_cloud[:, :3] = point_cloud[:, :3] @ rotation_matrix
    return point_cloud


def rotate_perturbation_point_cloud(point_cloud, angle_sigma=0.06, angle_clip=0.18):
    """Randomly perturb the point clouds by small rotations
     Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, rotated point cloud
    """
    angles = torch.clip(angle_sigma * torch.randn(3), -angle_clip, angle_clip)
    Rx = torch.Tensor(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = torch.Tensor(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = torch.Tensor(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    rotation_matrix = Rz @ Ry @ Rx
    point_cloud[:, :3] = point_cloud[:, :3] @ rotation_matrix
    return point_cloud


def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.
      Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, jittered point cloud
    """
    assert clip > 0
    N, _ = point_cloud.shape
    jitter = torch.clip(sigma * torch.randn(N, 3), -1 * clip, clip)
    point_cloud[:, :3] += jitter

    return point_cloud


def shift_point_cloud(point_cloud, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    point_cloud[:, :3] += shifts
    return point_cloud


def random_flip_point_cloud(point_cloud):
    """Randomly flip the point cloud. Flip is per point cloud.
    Input:
      Nx4 array, original point cloud
    Return:
      Nx4 array, flipped point cloud
    """
    if np.random.random() > 0.5:
        point_cloud[:, 0] *= -1
    if np.random.random() > 0.5:
        point_cloud[:, 1] *= -1
    return point_cloud


def random_scale_point_cloud(point_cloud, scale_low=0.95, scale_high=1.05):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high, 1)
    point_cloud[:, :3] *= scales
    return point_cloud
