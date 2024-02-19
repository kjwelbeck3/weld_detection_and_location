#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:29:08 2022

@author: mzx096
"""

import torch
from torch import nn
from torch.nn.functional import softmax
from torch_scatter import scatter_max
from sklearn.neighbors import NearestNeighbors


#tool: linear layers
def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU(),
        nn.BatchNorm1d(Ks[i])]
    return nn.Sequential(*linears)

def multi_layer_neural_network_fn_singular(Ks):
    ## Removes the batch norm compared to mutli_layer_neural_network_fn
    ## Updated:: Removed the ReLU on the final layer
    linears = []
    for i in range(1, len(Ks)-1):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU()]
    linears += [nn.Linear(Ks[-2], Ks[-1])]
    return nn.Sequential(*linears)

#tool: linear layers + output layers
## CHANGE: Different Default Ks and num_layers
def multi_layer_fc_fn(Ks=[300, 64, 32], num_classes=3, is_logits=False, num_layers=3):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
                nn.Linear(Ks[i-1], Ks[i]),
                nn.ReLU(),
                nn.BatchNorm1d(Ks[i])
                ]

    if is_logits:  ## [QUESTION]: should this not go through a sigmoid now
        linears += [
                nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
                nn.Linear(Ks[-1], num_classes),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes)
                ]
    return nn.Sequential(*linears)

## CHANGE: Different default Ks and num_layers
def prediction_fc_fn(Ks=[300, 64, 32], num_classes=3, is_logits=False, num_layers=3):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
                nn.Linear(Ks[i-1], Ks[i]),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm1d(Ks[i])
                ]

    if is_logits:
        linears += [
                nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
                nn.Linear(Ks[-1], num_classes),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm1d(num_classes)
                ]
    return nn.Sequential(*linears)

#tool: max aggregation function
def max_aggregation_fn(features, index, l):
    """
    Arg: features: N x dim
    index: N x 1, e.g.  [0,0,0,1,1,...l,l]
    l: lenght of keypoints

    returns keypoints' features : K x dim 
    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1]) # N x 64  ## S x 300  S==N
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1,0).contiguous() # len x 64  ## K x 300
    # set_features, argmax = scatter_max(features.permute(1,0), index.permute(1,0), out=set_features)
    set_features, argmax = scatter_max(features.permute(1,0), index.permute(1,0))
    set_features = set_features.permute(1,0)
    return set_features

def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
    """
     github.com/tensorflow/models/blob/master/\
         research/object_detection/core/losses.py
     Computer focal loss for binary classification
     Args:
       labels: A int32 tensor of shape [batch_size]. N x 1
       logits: A float32 tensor of shape [batch_size]. N x C
       alpha: A scalar for focal loss alpha hyper-parameter.
       If positive samples number > negtive samples number,
       alpha < 0.5 and vice versa.
       gamma: A scalar for focal loss gamma hyper-parameter.
     Returns:
       A tensor of the same shape as `labels`
     """
     
    prob = logits.sigmoid()    #### [CORRECTION] Should be softmax since multi-class classification; assuming the focal cross works this way
    labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=prob.shape[1])

    cross_ent = torch.clamp(logits, min=0) - logits * labels + torch.log(1+torch.exp(-torch.abs(logits)))
    prob_t = (labels*prob) + (1-labels) * (1-prob)
    modulating = torch.pow(1-prob_t, gamma)
    alpha_weight = (labels*alpha)+(1-labels)*(1-alpha)

    focal_cross_entropy = modulating * alpha_weight * cross_ent

    return focal_cross_entropy

def focal_loss_softmax(labels, logits, alpha=0.5, gamma=2):
    """
     github.com/tensorflow/models/blob/master/\
         research/object_detection/core/losses.py
     Computer focal loss for binary classification
     Args:
       labels: A int32 tensor of shape [batch_size]. N x 1
       logits: A float32 tensor of shape [batch_size]. N x C
       alpha: A scalar for focal loss alpha hyper-parameter.
       If positive samples number > negtive samples number,
       alpha < 0.5 and vice versa.
       gamma: A scalar for focal loss gamma hyper-parameter.
     Returns:
       A tensor of the same shape as `labels`
     """
    # print("from focal_loss_softmax")
    # probs = softmax(logits, dim=1)
    probs = logits
    labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=probs.shape[1])
    # print(f"logits shape: {logits.shape}")
    # print(f"labels shape: {labels.shape}")

    alphas = 1/labels.sum(dim=0)
    # print(f"alphas: {alphas}")

    alphas[torch.isinf(alphas)] = 0
    # print(f"alphas shape: {alphas.shape}")
    # print(f"alphas: {alphas}")

    # modulating = torch.pow(1-probs, gamma)
    pt = torch.sum(probs * labels, dim=1)
    focal_term = (1-pt)**gamma

    neg_log = -torch.log(probs)
    loss = torch.sum(neg_log*labels*alphas, dim=1)
    focal_cross_entropy = focal_term*loss
    # focal_cross_entropy = alphas * modulating * neg_log
    # print(f"focal_cross_entropy.shape: {focal_cross_entropy.shape}")
    # print(f"focal_cross_entropy: {focal_cross_entropy}")
    return focal_cross_entropy

class PointSetPooling(nn.Module):

    def __init__(self, point_MLP_depth_list=[3,32, 64, 128, 300], output_MLP_depth_list=[300, 300, 300]):
        super(PointSetPooling, self).__init__()

        Ks = list(point_MLP_depth_list)
        self.point_linears = multi_layer_neural_network_fn(Ks)
        
        Ks = list(output_MLP_depth_list)
        self.out_linears = multi_layer_neural_network_fn(Ks)
    
    def forward(self, 
            point_coordinates,
            keypoint_indices,
            set_indices):
        """
        Extract Edge Features from Points; 
        Pool Upstream features to Downstream features; 
        Update features per Keypoint.
        
        Args:
            point_coordinates: a [N, D] tensor. N is the number of points.
            D is the dimension of the coordinates.
            keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
            set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
            i.e. (i, j) indicates point[i] belongs to the point set created by
            grouping around keypoint[j].
        
        Returns: 
            dest_set_features: a [K, output_depth] tensor as the set feature.
                Output_depth depends on the feature extraction MLP outsput size.
        """

        ## Step 1: Prepare/Package edges' Source and Destination point coordinates
        src_set_coordinates = point_coordinates[set_indices[:, 0]]  ## S x 3
        dest_set_keypoint_indices = keypoint_indices[set_indices[:, 1]]  ## S x 1
        dest_set_coordinates = point_coordinates[dest_set_keypoint_indices[:, 0]]  ## S x 3 
        
        
        ## Step 2: Packages edges as free discplacment vectors  
        displacement_vectors = src_set_coordinates - dest_set_coordinates   ## S x 3
        
        
        ## Step 3: Extract/Expand features from Edges
        extracted_features = self.point_linears(displacement_vectors)  ## S x 300
        

        ## Step 4: Max Pool Features from Upstream layer to Downstream layer along edges
        set_features = max_aggregation_fn(extracted_features, set_indices[:, 1], keypoint_indices.shape[0])
        

        # Step 5: Update Downstream (ie Keypoint) features 
        dest_set_features = self.out_linears(set_features)
        


        return dest_set_features
    
    
class GraphNetAutoCenter(nn.Module):
    def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[300, 64, 3], edge_MLP_depth_list=[303, 300, 300], update_MLP_depth_list=[300, 300, 300]):
        super(GraphNetAutoCenter, self).__init__()
        self.auto_offset = auto_offset
        self.auto_offset_fn = multi_layer_neural_network_fn(auto_offset_MLP_depth_list)
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)


    def forward(self, input_vertex_features,
        input_vertex_coordinates,
        keypoint_indices,
        edges):
        """
        Extract Edge features from source features and displacements; 
        Pool Intra-layer source features to intra-layer destination features, then update; 
        Sum Destination and Source Features
        
        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
                M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
                vertices. D is the dimension of the coordinates.
            keypoint_indices: NOT_USED 
                leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
        
        Returns: 
            output_vertex_features: a [N, M] tensor. Updated layer vertex features.
        """

        ## Step 1: Prepare/Package edges' source features and point coordinates
        s_vertex_features = input_vertex_features[edges[:, 0]]
        s_vertex_coordinates = input_vertex_coordinates[edges[:, 0]]

        ## Step 2: Optionally generate offsets to point coordinates from point features
        if self.auto_offset:
            offset = self.auto_offset_fn(input_vertex_features)
            input_vertex_coordinates = input_vertex_coordinates + offset

        ## Step 3: Prepare edges' destination point coordinate, including the optional offset
        d_vertex_coordinates = input_vertex_coordinates[edges[:, 1]]

        ## Step 4: Free displacement vectors suffixed to source features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)
        
        ## Step 5: Extract source features from displacement
        edge_features = self.edge_feature_fn(edge_features)

        ## Step 6: Max Pool Source Features into Destination Features
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:,1], keypoint_indices.shape[0])
     
        ## Step 7: Update Destination Features
        update_features = self.update_fn(aggregated_edge_features)

        ## Step 8: Add Destination features to Orginal layer features
        output_vertex_features  = update_features + input_vertex_features
        return output_vertex_features    

class GraphNetAutoCenterToSingular(nn.Module):
    def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[300, 64, 3], edge_MLP_depth_list=[303, 300, 300], update_MLP_depth_list=[300, 300, 300]):
        super(GraphNetAutoCenterToSingular, self).__init__()
        self.auto_offset = auto_offset
        self.auto_offset_fn = multi_layer_neural_network_fn(auto_offset_MLP_depth_list)
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn_singular(update_MLP_depth_list)


    def forward(self, input_vertex_features,
        input_vertex_coordinates,
        keypoint_indices,
        edges):
        """apply one layer graph network on a graph. .
        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
        returns: a [N, M] tensor. Updated vertex features.
        """
        #print(f"input_vertex_features: {input_vertex_features.shape}")
        #print(f"input_vertex_coordinates: {input_vertex_coordinates.shape}")
        #print(f"keypoint_indices: {keypoint_indices.shape}")
        #print(f"edges: {edges.shape}")

        # Gather the source vertex of the edges
        s_vertex_features = input_vertex_features[edges[:, 0]]
        s_vertex_coordinates = input_vertex_coordinates[edges[:, 0]]

        if self.auto_offset:
            offset = self.auto_offset_fn(input_vertex_features)
            input_vertex_coordinates = input_vertex_coordinates + offset

        # Gather the destination vertex of the edges
        d_vertex_coordinates = input_vertex_coordinates[edges[:, 1]]

        # Prepare initial edge features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)
        
        # Extract edge features
        edge_features = self.edge_feature_fn(edge_features)
        #print("GraphNetAutoCenter,edge_features",edge_features.shape)
        # Aggregate edge features
        # aggregated_edge_features = max_aggregation_fn(edge_features, edges[:,1], len(keypoint_indices))
        # print(f"keypoint_indices: {keypoint_indices.shape}")

        ### WARNING CORRECTION
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:,1], keypoint_indices.shape[0])
        
        # #### DEBUG TOKEN
        # print("GraphNetAutoCenterToSingular,aggregated_edge_features",aggregated_edge_features.shape)
        
        # Update vertex features
        update_features = self.update_fn(aggregated_edge_features)
    
        # #### DEBUG TOKEN
        # print("GraphNetAutoCenterToSingular,update_features",update_features.shape)
        # print("GraphNetAutoCenterToSingular,input_vertex_features",input_vertex_features.shape)
        # print()
    
        # output_vertex_features  = update_features + input_vertex_features
        # return output_vertex_features   
        return update_features 



## CHANGE: Replaces ClassAwarePredictor; From Box encoding to instance segmentation 
class InstSegmtor(nn.Module):
    def __init__(self, num_classes, num_max_instance):
        super(InstSegmtor, self).__init__()
        self.cls_fn = multi_layer_fc_fn(Ks=[300, 64], num_layers=2, num_classes=num_classes, is_logits=True)
        
        self.num_classes = num_classes
        self.num_max_instance = num_max_instance


        self.seg_fn=prediction_fc_fn(Ks=[300, 64], num_layers=2, num_classes=num_max_instance, is_logits=True)
        
        
    def forward(self, features):
        logits = self.cls_fn(features)
        logits = softmax(logits, dim=1)
        
        inst_seg = self.seg_fn(features)#.unsqueeze(1)####DO I NEED UNSQUEEZE
        inst_seg = softmax(inst_seg, dim=1)
            

        
        return logits, inst_seg
    
    
class MultiLayerFastLocalGraphModelV2(nn.Module):

    ## CHANGE: Replaces ClassAwarePredictor with InstSegmtor
    def __init__(self, num_classes, max_instance_no, regularizer_type=None, \
            regularizer_kwargs=None, layer_configs=None, mode=None, graph_net_layers=3):
        super(MultiLayerFastLocalGraphModelV2, self).__init__()
        self.num_classes = num_classes
        self.max_instance_no = max_instance_no
        self.point_set_pooling = PointSetPooling()

        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):
            self.graph_nets.append(GraphNetAutoCenter())

        self.predictor = InstSegmtor(num_classes, max_instance_no)

    ## CHANGE: No longer expecting point normals
    # def forward(self, batch):
    def forward(self, vertex_coord_list, keypoint_indices_list, edges_list, cls_labels=None, inst_labels=None):
        # vertex_coord_list, keypoint_indices_list, edges_list, \
        #     cls_labels, inst_labels = batch

        point_coordinates, keypoint_indices, set_indices =  vertex_coord_list[0], keypoint_indices_list[0], edges_list[0]
        #print(point_features.shape)
        #print(point_coordinates.shape)
        #print(keypoint_indices.shape)
        #print(set_indices.shape)
        
        point_features = self.point_set_pooling(point_coordinates, keypoint_indices, set_indices)
        #print(point_features.shape)

        ##point_coordinates, keypoint_indices, set_indices = vertex_coord_list[1], keypoint_indices_list[1], edges_list[1]
        ##Does not work, size does not match, try to use keypoint ind 0
        point_coordinates, keypoint_indices, set_indices = vertex_coord_list[1], keypoint_indices_list[0], edges_list[1]
        for i, graph_net in enumerate(self.graph_nets):
            point_features = graph_net(point_features, point_coordinates, keypoint_indices, set_indices)
        cls_logits, inst_seg = self.predictor(point_features)
        return cls_logits, inst_seg

    def postprocess(self, logits):
        softmax = nn.Softmax(dim=1)
        prob = softmax(logits)
        return prob

    ## CHANGE: Takes in predictions and truths for sem and inst seg task; 
    ## not for encoding boxes; no need for huger_loss/nn.SmoothL1Loss
    def loss(self, logits, labels, inst_seg, gt_seg,  
            cls_loss_type="focal_sigmoid", seg_loss_type='focal_sigmoid', seg_loss_weight=1.0, cls_loss_weight=1.0):
        """Output loss value.
        Args:
            logits: [N, num_classes] tensor. The classification logits from
            predict method.
            labels: [N] tensor. The one hot class labels.
            inst_seg: [N, num_max_instance] tensor. The encoded
            instance segmentation from the predict method.
            gt_seg: [N, num_max_instance] tensor. The ground truth encoded
            bounding boxes.
            valid_box: [N] tensor. An indicator of whether the vertex is from
            an object of interest (whether it has a valid bounding box).
            cls_loss_type: string, the type of classification loss function.
            cls_loss_kwargs: dict, keyword args to the classifcation loss.
            seg_loss_type: string, the type of localization loss function.
            seg_loss_kwargs: dict, keyword args to the localization loss.
            loc_loss_weight: scalar, weight on localization loss.
            cls_loss_weight: scalar, weight on the classifcation loss.
        returns: a dict of cls_loss, loc_loss, reg_loss, num_endpoint,
        num_valid_endpoint. num_endpoint is the number of output vertices.
        num_valid_endpoint is the number of output vertices that have a valid
        bounding box. Those numbers are useful for weighting during batching.
        """
        """
        print(f"logits: {logits.shape}")
        print(f"labels: {labels.shape}")
        print(f"pred_box: {pred_box.shape}")
        print(f"gt_box: {gt_box.shape}")
        print(f"valid_box: {valid_box.shape}")
        """
        #print(f"pred_box: {pred_box.shape}")
        # point_cls_loss = focal_loss_sigmoid(labels,logits) # same shape as logits, N x C
        point_cls_loss = focal_loss_softmax(labels,logits) # same shape as logits, N x C
        num_endpoint = point_cls_loss.shape[0]
        cls_loss = cls_loss_weight * point_cls_loss.mean()
        
        # point_seg_loss = focal_loss_sigmoid(gt_seg,inst_seg) # same shape as logits, N x C
        point_seg_loss = focal_loss_softmax(gt_seg,inst_seg) # same shape as logits, N x C
        seg_loss = cls_loss_weight * point_seg_loss.mean()

        
        loss_dict = {}
        loss_dict['cls_loss'] = cls_loss

        params = torch.cat([x.view(-1) for x in self.parameters()])
        reg_loss = torch.mean(params.abs())

        loss_dict.update({'cls_loss': cls_loss, 'seg_loss': seg_loss,
            'reg_loss': reg_loss,
            'num_end_point': num_endpoint,
            })
        return loss_dict


    def accuracy(self, cls_preds, cls_labels, inst_preds, inst_labels):
        cls_preds = cls_preds.squeeze()
        cls_labels = cls_labels.squeeze()
        inst_preds = inst_preds.squeeze()
        inst_labels = inst_labels.squeeze()

        assert cls_preds.shape == cls_labels.shape
        assert inst_preds.shape == inst_labels.shape

        cls_true_count = int((cls_preds == cls_labels).sum())
        inst_true_count = int((inst_preds == inst_labels).sum())

        points_count = cls_labels.shape[0]
        return {'cls_accuracy': cls_true_count/points_count,
                'inst_accuracy': inst_true_count/points_count}
        
    
