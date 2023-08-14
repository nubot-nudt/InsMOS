#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Neng Wang
# Brief: visualizer based on open3D for moving object segmentation and bounding box 
# This file is covered by the LICENSE file in the root of this project.

import os
import sys
sys.path.append("..")
import yaml
import open3d as o3d
import pynput.keyboard as keyboard
from PIL import Image, ImageDraw, ImageFont
import copy
import numpy as np
from models.utils import Array_Index

def box_center_to_corner(bbox):
  """ create bounding boxes that can be used for open3d visualization
  Inputs:
    bbox: EKF bounding box [x, y, z, theta, l, w, h]
  Returns:
    corner_box: bounding box with 8 corner coordinates
  """
  translation = bbox[0:3]
  l, w, h = bbox[3], bbox[4], bbox[5]
  rotation = bbox[6]
  
  # Create a bounding box outline
  bounding_box = np.array([
    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
    [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])
  
  # Standard 3x3 rotation matrix around the Z axis
  rotation_matrix = np.array([
    [np.cos(rotation), -np.sin(rotation), 0.0],
    [np.sin(rotation), np.cos(rotation), 0.0],
    [0.0, 0.0, 1.0]])
  
  # Repeat the [x, y, z] eight times
  eight_points = np.tile(translation, (8, 1))
  
  # Translate the rotated bounding box by the
  # original center position to obtain the final box
  corner_box = np.dot(
    rotation_matrix, bounding_box) + eight_points.transpose()
  
  return corner_box.transpose()
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def load_vertex(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
    the KITTI dataset.
    Args:
      scan_path: the (full) filename of the scan file
    Returns:
      A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex


def load_files(folder):
  """ Load all files in a folder and sort.
  """
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths

def load_labels(label_path):
  label = np.fromfile(label_path, dtype=np.uint32)
  label = label.reshape((-1))
  #print("label:",label)
  sem_label = label & 0xFFFF  # semantic label in lower half
  inst_label = label >> 16  # instance id in upper half

  # sanity check
  assert ((sem_label + (inst_label << 16) == label).all())
  
  return sem_label, inst_label

def load_bbox(bbox_path):
    preb_bbox = np.load(bbox_path,allow_pickle=True)
    preb_bbox = preb_bbox.item()
    return preb_bbox


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
  
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            color = self.colors
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

class vis_mos_results:
  """ LiDAR moving object segmentation results (LiDAR-MOS) visualizer
  Keyboard navigation:
    n: play next
    b: play back
    esc or q: exits
  """
  def __init__(self, config):
    # specify paths
    seq = str(config['seq']).zfill(2)
    dataset_root = config['dataset_root']
    prediction_root = config['prediction_root']
    prediction_bbox_root = config['prediction_bbox_root']
    
    # specify folders
    scan_folder = os.path.join(dataset_root, 'sequences',seq, 'velodyne')
    gt_folder = os.path.join(dataset_root,'sequences', seq, 'labels')
    prediction_folder = os.path.join(prediction_root, 'sequences', seq, 'predictions')
    gt_folder = prediction_folder
    prediction_bbox_folder = os.path.join(prediction_bbox_root,'sequences', seq, 'predictions')

    self.flag = False
    self.bbox_line_list = []
    self.bbox_line_list_old = []
    # load files
    self.scan_files = load_files(scan_folder)
    self.gt_paths = load_files(gt_folder)
    self.predictions_files = load_files(prediction_folder)
    self.predictions_bbox_files = load_files(prediction_bbox_folder)
    
    # init frame
    self.current_points = load_vertex(self.scan_files[0])[:, :3]
    self.current_preds,_= load_labels(self.predictions_files[0])
    self.current_gt, _ = load_labels(self.gt_paths[0])
    self.bbox = load_bbox(self.predictions_bbox_files[0])
    
    self.pcd = o3d.geometry.PointCloud()
    self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
    self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
    colors = np.array(self.pcd.colors)
    tp = (self.current_preds > 200) & (self.current_gt > 200)
    fp = (self.current_preds > 200) & (self.current_gt < 200)
    fn = (self.current_preds < 200) & (self.current_gt > 200)
    print("tp:",tp)
    print("fp:",fp)
    print("fn:",fn)

  
    colors[tp] = [0, 1, 0]
    colors[fp] = [1, 0, 0]
    colors[fn] = [0, 0, 1]
  
    self.pcd.colors = o3d.utility.Vector3dVector(colors)
  
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-45, -45, -5),
                                               max_bound=(45, 45, 5))
    self.pcd = self.pcd.crop(bbox)  # set view area

    # init visualizer
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window(window_name='pcd', width=1440, height=1080)
    render_option: o3d.visualization.RenderOption = self.vis.get_render_option()	# set render
    render_option.background_color = np.array([0, 0, 0])	# set background
    render_option.point_size = 2.0
    # ctr = self.vis.get_view_control()
    # param = o3d.io.read_pinhole_camera_parameters("render.json")
    self.vis.add_geometry(self.pcd)
  
    # init keyboard controller
    key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
    key_listener.start()
    
    # init frame index
    self.frame_idx = 0
    self.current_frame_idx = self.frame_idx
    self.num_frames = len(self.scan_files)

  def on_press(self, key):
    try:
      #a=0
      self.flag = True
      if key.char == 'q':
        try:
          sys.exit(0)
        except SystemExit:
          os._exit(0)
        
      if key.char == 'n':
        if self.frame_idx < self.num_frames - 1:
          self.frame_idx += 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds,_= load_labels(self.predictions_files[self.frame_idx])
          self.bbox = load_bbox(self.predictions_bbox_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('Reach the end of this sequence!')
          
      if key.char == 'b':
        if self.frame_idx > 1:
          self.frame_idx -= 1
          self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
          self.current_preds,_= load_labels(self.predictions_files[self.frame_idx])
          self.bbox = load_bbox(self.predictions_bbox_files[self.frame_idx])
          self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
          print("frame index:", self.frame_idx)
        else:
          print('At the start at this sequence!')
          
    except AttributeError:
      print('special key {0} pressed'.format(key))
      
  def on_release(self, key):
    try:
      #a=0
      if key.char == 'n':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds,_= load_labels(self.predictions_files[self.frame_idx])
        self.bbox = load_bbox(self.predictions_bbox_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
    
      if key.char == 'b':
        self.current_points = load_vertex(self.scan_files[self.frame_idx])[:, :3]
        self.current_preds,_ = load_labels(self.predictions_files[self.frame_idx])
        self.bbox = load_bbox(self.predictions_bbox_files[self.frame_idx])
        self.current_gt, _ = load_labels(self.gt_paths[self.frame_idx])
        
    except AttributeError:
      print('special key {0} pressed'.format(key))
  
  def run(self):
    current_points = copy.deepcopy(self.current_points)
    current_preds = copy.deepcopy(self.current_preds)
    current_gt = copy.deepcopy(self.current_gt)
    current_bbox = copy.deepcopy(self.bbox)

    if (len(current_points) == len(current_preds)) \
        and (len(current_points) == len(current_gt)) \
        and (len(current_preds) == len(current_gt)):

      for bounding_box_idx in range(0,len(current_bbox['pred_boxes'])):
        if (current_bbox['pred_scores'][bounding_box_idx]<0.5 and current_bbox['pred_labels'][bounding_box_idx]==1) or \
          (current_bbox['pred_scores'][bounding_box_idx]<0.2 and current_bbox['pred_labels'][bounding_box_idx]!=1):
          current_bbox['pred_boxes'][bounding_box_idx][:] = 0

      boxes = np.concatenate((current_bbox['pred_boxes'],current_bbox['pred_labels'].reshape(-1,1)),axis=1)
      features =np.zeros((current_points.shape[0],3),dtype=int)
      index = Array_Index.find_point_in_instance_bbox_with_yaw(current_points,boxes,features,0.2)
      
      colors = np.zeros([current_points.shape[0], 3])
      colors[:] = [0.5,0.5,0.5]
      colors[index[:,0]>0] = [0/255, 255/255, 255/255]
      colors[index[:,1]>0] = [0/255, 255/255, 255/255]

      if self.current_frame_idx!= self.frame_idx:
        self.current_frame_idx = self.frame_idx
        self.pcd.points = o3d.utility.Vector3dVector(current_points)
        colors[current_preds==251] = [1,0,0]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.clear_geometries()
        # ctr = self.vis.get_view_control()
        # param = o3d.io.read_pinhole_camera_parameters("render.json")
        self.vis.add_geometry(self.pcd)

        # print("bbox:",current_bbox['pred_boxes'])
        if len(current_bbox['pred_boxes'])!=0:
          for bounding_box_idx in range(len(current_bbox['pred_boxes'])):
              if 1:
                inst_lable = current_bbox['pred_labels'][bounding_box_idx]
                bbox_corner = box_center_to_corner(current_bbox['pred_boxes'][bounding_box_idx])
                if inst_lable==1:
                    color=[0/255, 255/255, 0/255]
                elif inst_lable==2:
                    color=[0/255,0/255,255/255]
                elif inst_lable==3:
                    color=[255/255, 255/255, 0/255]
                
                points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
                        [0, 1, 1], [1, 1, 1]]
                lines = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                [0, 4], [1, 5], [2, 6], [3, 7]])
                colors = [[1, 0, 0] for i in range(len(lines))]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbox_corner)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                # Create Line Mesh 1
                line_mesh1 = LineMesh(bbox_corner, lines, color, radius=0.03)
                line_mesh1_geoms = line_mesh1.cylinder_segments


                # ctr = self.vis.get_view_control()
                # param = o3d.io.read_pinhole_camera_parameters("render.json")
                line_mesh1.add_line(self.vis)

      self.vis.poll_events()
      self.vis.update_renderer()
  
  
if __name__ == '__main__':
  # load config file
  config_filename = 'dataset_root.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))

  # init the mos visualizer
  visualizer = vis_mos_results(config)
  
  # run the visualizer
  while True:
    visualizer.run()