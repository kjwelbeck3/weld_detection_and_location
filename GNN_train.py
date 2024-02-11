#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:32:24 2022

@author: mzx096
"""
import os
import torch
from datetime import datetime
import random
from dataset import MyDataset
from Instseg_model import MultiLayerFastLocalGraphModelV2
import torch.utils.data as data_utils

cuda = False
print("Cuda available", torch.cuda.is_available())
print("Cuda ", cuda)

LR = 1e-4
epoches = 50

model_kwargs = {
    'layer_configs': [
        {
            'graph_level': 0,
            'kwargs': {'output_MLP_activation_type': 'ReLU',
                        'output_MLP_depth_list': [256, 256],
                        'output_MLP_normalization_type': 'NONE',
                        'point_MLP_activation_type': 'ReLU',
                        'point_MLP_depth_list': [32, 128, 256],
                        'point_MLP_normalization_type': 'NONE'},
           'scope': 'layer1',
           'type': 'scatter_max_point_set_pooling'},
        {
            'graph_level': 1,
            'kwargs': {'auto_offset': True,
                        'auto_offset_MLP_depth_list': [64, 3],
                        'auto_offset_MLP_feature_activation_type': 'ReLU',
                        'auto_offset_MLP_normalization_type': 'NONE',
                        'edge_MLP_activation_type': 'ReLU',
                        'edge_MLP_depth_list': [256, 256],
                        'edge_MLP_normalization_type': 'NONE',
                        'update_MLP_activation_type': 'ReLU',
                        'update_MLP_depth_list': [256, 256],
                        'update_MLP_normalization_type': 'NONE'},
           'scope': 'layer2',
           'type': 'scatter_max_graph_auto_center_net'},
        {
            'graph_level': 1,
            'kwargs': {'auto_offset': True,
                        'auto_offset_MLP_depth_list': [64, 3],
                        'auto_offset_MLP_feature_activation_type': 'ReLU',
                        'auto_offset_MLP_normalization_type': 'NONE',
                        'edge_MLP_activation_type': 'ReLU',
                        'edge_MLP_depth_list': [256, 256],
                        'edge_MLP_normalization_type': 'NONE',
                        'update_MLP_activation_type': 'ReLU',
                        'update_MLP_depth_list': [256, 256],
                        'update_MLP_normalization_type': 'NONE'},
            'scope': 'layer3',
            'type': 'scatter_max_graph_auto_center_net'},
        {
            'graph_level': 1,
            'kwargs': {'auto_offset': True,
                        'auto_offset_MLP_depth_list': [64, 3],
                        'auto_offset_MLP_feature_activation_type': 'ReLU',
                        'auto_offset_MLP_normalization_type': 'NONE',
                        'edge_MLP_activation_type': 'ReLU',
                        'edge_MLP_depth_list': [256, 256],
                        'edge_MLP_normalization_type': 'NONE',
                        'update_MLP_activation_type': 'ReLU',
                        'update_MLP_depth_list': [256, 256],
                        'update_MLP_normalization_type': 'NONE'},
            'scope': 'layer4',
            'type': 'scatter_max_graph_auto_center_net'},
          {
              'graph_level': 1,
              'kwargs': {'activation_type': 'ReLU', 'normalization_type': 'NONE'},
               'scope': 'output',
               'type': 'classaware_predictor'}
    ],
     'regularizer_kwargs': {'scale': 5e-07},
     'regularizer_type': 'l1'}


model = MultiLayerFastLocalGraphModelV2(num_classes=3,
            max_instance_no=6, mode='train',**model_kwargs)

if cuda:         
    model = model.cuda()


dataset_root = "./data/"
data_list = []
for i in range(1,301):
    data_list.append([
        dataset_root+'bin_file/photoneo1_a_ply'+ str(i)+'.bin',
        dataset_root+'gt/photoneo1_a_gt_'+ str(i)+'.txt'
    ])
    data_list.append([
        dataset_root+'bin_file/photoneo1_b_ply'+ str(i)+'.bin',
        dataset_root+'gt/photoneo1_b_gt_'+ str(i)+'.txt'
    ])
    data_list.append([
        dataset_root+'bin_file/photoneo2_a_ply'+ str(i)+'.bin',
        dataset_root+'gt/photoneo2_a_gt_'+ str(i)+'.txt'
    ])
    data_list.append([
        dataset_root+'bin_file/photoneo2_b_ply'+ str(i)+'.bin',
        dataset_root+'gt/photoneo2_b_gt_'+ str(i)+'.txt'
    ])
    

random.shuffle(data_list)
train_set = data_list[:1100]
test_set = data_list[1100:]    

with open(r'./trainset.txt', 'w') as fp:
    for item in train_set:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing training items')
    
with open(r'./testset.txt', 'w') as fp:
    for item in test_set:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing training items')

train_data = MyDataset(dataset=train_set)
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=1)#, shuffle=True, num_workers=0)

test_data=MyDataset(dataset=test_set)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# train_root = '/home/mzx096/Documents/Instance_seg/Train_1214'
# root_2 = train_root+'/model/'
# root_3 = train_root+'/LOG_'
# root_4 = train_root+'/valLOG_'
# #os.mkdir(root_2)
# #os.mkdir(root_4)
# #os.mkdir(root_3)

train_root = '.'
# root_2 = train_root+'/model/'
# root_3 = train_root+'/LOG_'
# root_4 = train_root+'/valLOG_'

root_2 = train_root+'/_model/'
if not os.path.isdir(root_2):
    os.mkdir(root_2)
root_3 = train_root+'/_LOG_'
if not os.path.isdir(root_3):
    os.mkdir(root_3)
root_4 = train_root+'/_valLOG_'
if not os.path.isdir(root_4):
    os.mkdir(root_4)

# dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H:%M:%S')
dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
fileOut=open(root_3+'/LOG_'+dataString+'_','w+')
fileOut2=open(root_4+'/valLOG_'+dataString+'_','w+')
fileOut.write('Epoch:   Step:    Loss:        Val_Accu :\n')
fileOut2.write('Epoch:   Step:    Loss:        Val_Accu :\n')
fileOut.close()
fileOut2.close()

for epoch in range(1,epoches):
    recalls_list, precisions_list, mAP_list = {}, {}, {}
    for i in range(3): 
        recalls_list[i], precisions_list[i], mAP_list[i] = [], [], []
        
    ## CHANGED - Previously packaged with point_normals
    for step, (vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, inst_labels) in enumerate(train_loader):
        
        batch = (vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, inst_labels)
        new_batch = []
        #k=0
        for item in batch:
            #print(k)

            if not isinstance(item, torch.Tensor):
                item = [torch.squeeze(x,0).cuda() if cuda else torch.squeeze(x, 0) for x in item]
                    
            else: 
                #print(item.shape)
                item = torch.squeeze(item,0).cuda() if cuda else torch.squeeze(item, 0)
                #print(item.shape)
                
            new_batch += [item]
            #k+=1
            

        vertex_coord_list, keypoint_indices_list, edges_list, \
                cls_labels, inst_labels = new_batch
    
        batch = new_batch
        logits, inst_seg = model(batch, is_training=True) ## CHANGED - prev: logits, box_encoding
        
        cls_predictions = torch.argmax(logits, dim=1)
        seg_predictions = torch.argmax(inst_seg, dim=1) 
        
        ## CHANGED Model's Loss changed to replace regression loss with mullt-classification loss
        loss_dict = model.loss(logits, cls_labels, inst_seg, inst_labels,cls_loss_weight=1)
        
        t_cls_loss, t_seg_loss, t_reg_loss = loss_dict['cls_loss'], loss_dict['seg_loss'], loss_dict['reg_loss']
        t_total_loss = t_cls_loss + 10*t_seg_loss + t_reg_loss#t_cls_loss + t_seg_loss + t_reg_loss
        optimizer.zero_grad()
        t_total_loss.backward()
        optimizer.step()
        
        if step%20 ==0:
        	print(epoch,  step, loss_dict['seg_loss'].data.item(), loss_dict['cls_loss'].data.item(), loss_dict['reg_loss'].data.item())
        fileOut=open(root_3+'/LOG_'+dataString+'_','a+')
        fileOut.write(str(epoch)+'   '+str(step)+'   '+str(loss_dict['seg_loss'].data.item())+"  "+ \
                          str(loss_dict['cls_loss'].data.item())+"  "+ \
                          str(loss_dict['reg_loss'].data.item())+'\n')
        #fileOut.write(loss_dict)
        #fileOut.write('\n')
        fileOut.close()
        
    if epoch%10 == 9:
        PATH = root_2 + 'param_1212_updated_loss_06_64_192_' + str(epoch) + '_' + str(step)
        torch.save(model.state_dict(), PATH)
        print('finished saving checkpoints')
