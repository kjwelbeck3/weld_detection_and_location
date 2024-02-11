#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:33:59 2022

@author: mzx096
"""

import numpy as np
import csv
import open3d as o3d
import torch.utils.data as data_utils
import torch
from graph_generation import gen_multi_level_local_graph_v3
from ply_reader import read_ply_header

## Sample Graph Definition to Transform Point Cloud to Graph
## For constructing layers of points, keypoints, and the inter- and intra-layer edges
graph_gen_kwargs = {
    'add_rnd3d': True,
    'base_voxel_size': 0.8,
    'downsample_method': 'random',
    'level_configs': [
        {'graph_gen_kwargs': {'num_neighbors': 64, 'radius': 0.4},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 0,
         'graph_scale': 1},
        {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 1.2},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 1,
         'graph_scale': 1}]
}

def gtloader(filepath):
    """
    Parses a segmentation label file
    Content of example label file representing 4 welds:
        -77.42,  11.06, [
        12.38,  12.26, [
        77.00,  13.22, ]
        87.60,  13.22, ]

    Returns:
        datalist: a list of lists 
            eg [[-77.42, 11.06, 1], 
                [12.38, 12.26, 1], 
                [77.00, 13.22, 2],
                []87.60, 13.22, 2]]
    """

    my_file = open(filepath, 'r')
    data = my_file.read()
    my_file.close()
    content_list = data.split("\n")
    
    datalist = []

    for item in content_list:
        if item:
            
            item_ = item.split(",")
            if item_[2].strip() == '[':
                item_[2] = 1
            else:
                item_[2] = 2
            datalist.append(item_)
    return datalist


def pcloader(filepath):
    """"Parent utility to handle loading point clouds from .csv, .bin and .ply formats"""
    pcd_p, offset = None, None

    if filepath.split(".")[-1] == "csv":
        pcd_p, offset = csvloader(filepath)

    elif filepath.split(".")[-1] == "bin":
        pcd_p, offset = binloader(filepath)

    elif filepath.split(".")[-1] == "ply":
        pcd_p, offset = plyloader(filepath)
    
    else:
        assert False , "The point cloud loading utility was expecting .csv/.bin/.ply file formats."

    return pcd_p, offset

def binloader(filepath):
    """Handles loading point clouds from .bin file format"""
    
    fid = open(filepath, 'rb')
    height = int.from_bytes(fid.read(8), "little", signed = 'true')
    length = int.from_bytes(fid.read(8), "little", signed = 'true')
    
    # First read to the point of the first offset.
    s1 = length *height
    
    points = np.ndarray((s1, 3), np.double, buffer = fid.read(24*s1),strides=(24, 8))
    
    ## Remove very distant points
    m = ~np.any(np.abs(points) > 1000.0, axis=1)
    points = points[m, :]

    ## Keep only nonzero points
    #points[:,2] = points[:, 2]*(-1)
    points[~np.all(points == 0.0, axis=1),:]
    offset = np.array([0,0,0])
    # offset = np.array([np.min(points[:,0]),np.min(points[:,1]),np.mean(points[:,2])])
    points = points-offset
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.2)
    
    pcd_p = np.asarray(pcd.points)
    return pcd_p, offset

def csvloader(filepath):
    """Handles loading point clouds from .csv file format"""

    # assert filepath.split(".")[-1] == "csv", "csvloader function was expecting a .csv file"
    
    ## DEBUG print statements
    # print(f"Loading filepath: {filepath}")

    reader = csv.reader(open(filepath))

    ## Skipping pass header info to the points data
    for row in reader:
        if row and row[0] == 'Y\X':
            break

    ## grouping x,y,z of each point from the file
    ## file is structured as 
    ##      x's as the header row ie horiz axis
    ##      y's as the header col ie vert axis
    ##      z's as the row-col intersections
    xx = [float(s) for s in row[1:]]
    xyz = []
    for row in reader:
        if row and row[0] == 'End':
            break
        y = float(row[0])
        zz = [float(s) if s else None for s in row[1:]]
        for i,z in enumerate(zz):
            if z is not None:
                xyz.append([xx[i], y, -z])
    points = np.array(xyz)

    # ## DEBUG print statement
    # print(f"full_cloud points shape: {points.shape}")
    
    ''' Remove very distant points. '''
    m = ~np.any(np.abs(points) > 1000.0, axis=1)
    points = points[m, :]

    # ## DEBUG print statement
    # print(f"outliers_removed points shape: {points.shape}")
    
    ''' Keep only nonzero points. '''
    points = points[~np.all(points == 0.0, axis=1),:] 

    # ## DEBUG print statement
    # print(f"non-zeroes_removed points shape: {points.shape}")
    

    ## zeroing location at the smaller vals along the x and y axes and at the mid height
    # offset = np.array([np.min(points[:,0]),np.min(points[:,1]),np.mean(points[:,2])])
    offset = np.array([0,0,0])
    points = points-offset
    
    ## Using the open3D PointCloud class to further preprocess matrix of points
    ## Downsampling, computing point normals, 
    
    ## Converts numpy array to open3d object for downsampling util
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.2)

    pcd_p = np.asarray(pcd.points)
    return pcd_p, offset

def plyloader(filepath):
    """Handles loading point clouds from .ply file format"""
    fid = open(filepath, 'rb')
    header = read_ply_header(fid)

    # First read to the point of the first offset.
    s1 = header.vertex_size
    s2 = header.y_index - header.x_index
    assert s2 == 4, 'Vertex coordinates not floats'
    assert header.z_index - header.y_index == 4, 'Vertex coordinates not floats'
    # print(header.x_index)
    # print(header.vertex_size)
    # print(header.vertex_ct)
    if header.x_index > 0:
        fid.read(header.x_index)
    points = np.ndarray((header.vertex_ct, 3), np.float32, 
                           buffer = fid.read(s1*header.vertex_ct),
                           strides=(header.vertex_size, s2))

    ## Remove very distant points
    m = ~np.any(np.abs(points) > 1000.0, axis=1)
    points = points[m, :]
    
    ## Keep only nonzero points
    points = points[~np.all(points == 0.0, axis=1),:]
    offset = np.array([0,0,0])
    # offset = np.array([np.min(points[:,0]),np.min(points[:,1]),np.mean(points[:,2])])
    points = points-offset
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.2)
    
    pcd_p = np.asarray(pcd.points)
    return pcd_p, offset

def assign_weld_region(last_layer_v,label,offset):
    """
    Overlays 2D template of weld type at weld location to extract 
    subset of point cloud representing the weld 
    Args:
        last_layer_v: 
            [Nx3] np.ndarray representing N vertices/coordinates of the point cloud 
        label: 
            [1x3] np.ndarray representing weld location and type ie [[x, y, 1 or 2]]
        offset:
            [1x2] np.ndarray to adjust the origin of point cloud

    Returns:
        mask:
            [Nx1] True/False np.ndarray for per-point indication of 2D template overlay 
    """

    xc1 = np.array([float(label[0])-offset[0], float(label[1])-offset[1]+7.53])
    xc2 = np.array([float(label[0])-offset[0], float(label[1])-offset[1]-7.53])
    
    mask1 = np.logical_or(np.linalg.norm(last_layer_v[:,:2]-xc1, axis=1)<=3.8, 
                          np.linalg.norm(last_layer_v[:,:2]-xc2, axis=1)<=3.8)
    
    mask2 = np.logical_and(last_layer_v[:, 1] >= float(label[1])-offset[1]-7.5, last_layer_v[:, 1] <= float(label[1])-offset[1]+7.5)
    if float(label[2]) == 1:
        
        mask3 = np.logical_and(last_layer_v[:, 0] >= float(label[0])-offset[0]-3.8, last_layer_v[:, 0] <= float(label[0])-offset[0]-1.2)
    else:
        mask3 = np.logical_and(last_layer_v[:, 0] <= float(label[0])-offset[0]+3.8, last_layer_v[:, 0] >= float(label[0])-offset[0]+1.2)
        
    mask2_ = np.logical_and(mask2,mask3)
    mask = np.logical_or(mask1,mask2_)
    return mask
    
    
class MyDataset(data_utils.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, loader=pcloader):
        self.imgs = dataset
        self.loader = loader
        
    def __getitem__(self, index):
        point_cloud_data, gt_data = self.imgs[index]
        pcd_p,  offset = self.loader(point_cloud_data)
        #pcd_p = torch.from_numpy(pcd_p)
        #pcd_n = torch.from_numpy(pcd_n)
        
        labellist = gtloader(gt_data)
        
        # ## DEBUG print statements
        # print("labellist",labellist)
        
        ## Generates multi-level graphs by 
        ## For Nodes/Vertices
        ## Listing out all points in first layer, 
        ## then randomized selections after voxelizing into second layer, 
        ## then repeating voxelized Selection into third layer
        ## For the second and third layers, keypoints_indices as which of the preceeding layer positions are copied into the next layer
        ## For Edges,
        ## Source to Destination pairing on both sets of layer pairs
        ## ie first and second edges; second and third edges.

        vertex_coord_list, keypoint_indices_list, edges_list = \
        gen_multi_level_local_graph_v3(pcd_p,0.6,graph_gen_kwargs['level_configs'])
        
        last_layer_v = vertex_coord_list[-1]
        Nv = vertex_coord_list[-1].shape[0]
        cls_labels = np.zeros((Nv, 1), dtype=np.int64)
        seg_labels = np.zeros((Nv, 1), dtype=np.int64)
        
        ## Identified the graphs and graph levels, labeling each final vertex/node with instance and type 
        instance_no = 0
        for label in labellist:
            instance_no +=1
            xcoor,ycoor,obj_cls = float(label[0]),float(label[1]),label[2]
            mask = assign_weld_region(last_layer_v,label,offset)
            cls_labels[mask, :] = obj_cls
            seg_labels[mask, :] = instance_no
            
            
        
        ##type casting
        #pcd_n = pcd_n.astype(np.float32)   
        vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
        keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
        edges_list = [e.astype(np.int32) for e in edges_list]
        cls_labels = cls_labels.astype(np.int32)
        seg_labels = seg_labels.astype(np.float32)
        
        #numpy array to tensor
        #pcd_n = torch.from_numpy(pcd_n)
        vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
        keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
        edges_list = [torch.from_numpy(item).long() for item in edges_list]
        cls_labels = torch.from_numpy(cls_labels)
        seg_labels = torch.from_numpy(seg_labels)
        #boxes_3d = torch.unsqueeze(boxes_3d,1)
        return  vertex_coord_list, keypoint_indices_list, edges_list, cls_labels, seg_labels
        
    def __len__(self):
        
        return len(self.imgs)    
    

