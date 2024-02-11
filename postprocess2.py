import sys
import os
import numpy as np
import torch
from torch import nn
from datetime import datetime
import random
from dataset import MyDataset
from graph_generation import gen_multi_level_local_graph_v3
from Instseg_model import MultiLayerFastLocalGraphModelV2
import torch.utils.data as data_utils
import matplotlib.pyplot
from dataset import gtloader, pcloader, assign_weld_region
import time

cuda = True
print("Cuda available :", torch.cuda.is_available())
print("cuda ", cuda and torch.cuda.is_available())

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


model = MultiLayerFastLocalGraphModelV2(num_classes=3,
            max_instance_no=7)

if cuda:            
	model = model.cuda()


# dataset_root = "/home/mzx096/Documents/dataset/"
dataset_root = "./_data/"
data_list = []


with open(dataset_root+"train.txt", "r") as traintext:
        samples = traintext.readlines()
        for sample in samples:
            # train_set.append(eval(sample))
            _sample = [dataset_root + a[2:] for a in eval(sample)]
            data_list.append(_sample)

data_list = data_list[-30:]

# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(5)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(5)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(26)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(26)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(27)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(27)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_b_ply'+ str(5)+'.bin',
#     dataset_root+'gt/photoneo1_b_gt_'+ str(5)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_b_ply'+ str(26)+'.bin',
#     dataset_root+'gt/photoneo1_b_gt_'+ str(26)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_b_ply'+ str(27)+'.bin',
#     dataset_root+'gt/photoneo1_b_gt_'+ str(27)+'.txt'])
    
# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(45)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(45)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(256)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(256)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo1_a_ply'+ str(127)+'.bin',
#     dataset_root+'gt/photoneo1_a_gt_'+ str(127)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo2_b_ply'+ str(65)+'.bin',
#     dataset_root+'gt/photoneo2_b_gt_'+ str(65)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo2_b_ply'+ str(30)+'.bin',
#     dataset_root+'gt/photoneo2_b_gt_'+ str(30)+'.txt'])
# data_list.append([
#     dataset_root+'bin_file/photoneo2_b_ply'+ str(180)+'.bin',
#     dataset_root+'gt/photoneo2_b_gt_'+ str(180)+'.txt'])
# data_list.append([dataset_root+'bin_file/photoneo2_b_ply24.bin', dataset_root+'gt/photoneo2_b_gt_24.txt']   ) 

print(data_list)

## Check that teach of these locations exit
bin_exists_count = 0
samples_count = 0
txt_exists_count = 0

for bin, txt in data_list:
    samples_count+=1    
    if os.path.isfile(bin):
        bin_exists_count+=1
    else:
        print(f"Does not exist: {bin}")
    if os.path.isfile(txt):
        txt_exists_count+=1
    else:
        print(f"Does not exist: {txt}")

print(f"samples_count: {samples_count}")
print(f"bin_exists_count: {bin_exists_count}")
print(f"txt_exists_count: {txt_exists_count}")

# train_root = '/home/mzx096/Documents/GNN/Train1117'
# root_2 = train_root+'/model/'
# root_3 = train_root+'/results'
# root_4 = root_3+'/results_1117'
# #os.mkdir(root_4)

postprocess_dir = "./results/_postprocess2/"
if os.path.isdir(postprocess_dir):
	import shutil
	shutil.rmtree(postprocess_dir)
os.mkdir(postprocess_dir)

logfile = open('./results/_postprocess2/print_statements.txt', 'w')

# model.load_state_dict(torch.load('./model/param_1212_updated_loss_06_64_192_49_1099'))
# model.load_state_dict(torch.load('./_model/train1/2023_06_03_18_58_57/params_epoch49_2023_06_03_23_22_28.pt'))
model.load_state_dict(torch.load('./_model/train1-fix1/2023_06_07_17_27_40/params_epoch49_2023_06_08_01_58_31.pt'))

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
for i in range(samples_count):
	a = time.time()
	pointxyz, offset = pcloader(data_list[i][0])  ## possible remove offset so as not to rezero cloud points
	vertex_coord_list, keypoint_indices_list, edges_list = \
	gen_multi_level_local_graph_v3(pointxyz,0.6,graph_gen_kwargs['level_configs'])
	labellist = gtloader(data_list[i][1])

	last_layer_v = vertex_coord_list[-1]
	Nv = vertex_coord_list[-1].shape[0]
	cls_labels = np.zeros((Nv, 1), dtype=np.int64)
	seg_labels = np.zeros((Nv, 1), dtype=np.int64)
	
	instance_no = 0
	for label in labellist:
	
	    xcoor,ycoor,obj_cls = float(label[0]),float(label[1]),float(label[2])
	    instance_no +=1
	    mask = assign_weld_region(last_layer_v,label,offset)
	    cls_labels[mask, :] = obj_cls
	    seg_labels[mask, :] = instance_no
	   
	    
	#pointn = pointn.astype(np.float32)   
	vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
	keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
	edges_list = [e.astype(np.int32) for e in edges_list]
	cls_labels = cls_labels.astype(np.int32)
	seg_labels = seg_labels.astype(np.float32)

	#numpy array to tensor
	vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
	keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
	edges_list = [torch.from_numpy(item).long() for item in edges_list]
	#pointn = torch.from_numpy(pointn)
	#vertex_coord_list = torch.from_numpy(vertex_coord_list)
	#keypoint_indices_list = torch.from_numpy(keypoint_indices_list)
	#edges_list = torch.from_numpy(edges_list)
	cls_labels = torch.from_numpy(cls_labels)
	seg_labels = torch.from_numpy(seg_labels)
	
	batch = (vertex_coord_list, keypoint_indices_list, edges_list,
		cls_labels, seg_labels)
	new_batch = []

	if cuda:
		for item in batch:
			

			if not isinstance(item, torch.Tensor):
					item = [x.cuda() for x in item]
			else: 
					item = item.cuda()
			new_batch += [item]
			
			
		vertex_coord_list, keypoint_indices_list, edges_list, \
			cls_labels, seg_labels = new_batch

		batch = new_batch

	logits, inst_seg = None, None
	with torch.no_grad():
		logits, inst_seg = model(*batch)#, is_training=False)
	
	b = time.time()
	print(b-a)

	predictions = torch.argmax(logits, dim=1)
	segmentations = torch.argmax(inst_seg, dim=1)
	
	logfile.write(f'{i}\n')
	logfile.write('logits:\n')
	logfile.write(f"{logits}")
	logfile.write('inst_seg:\n')
	logfile.write(f"{inst_seg}")
	logfile.write('predictions:\n')
	logfile.write(f"{predictions}")
	logfile.write('segmentation:\n')
	logfile.write(f"{segmentations}")
	logfile.write("####\n\n")

	################################################################
	#huger_loss = nn.SmoothL1Loss(reduction="none")
	#all_loc_loss = huger_loss(box_encoding, boxes_3d.squeeze())
	#loss_weight_mask = valid_boxes.squeeze(1)*9+1
	#loss_weight_mask = torch.cat((loss_weight_mask, loss_weight_mask), 1)
	#print(loss_weight_mask.shape,all_loc_loss.shape,valid_box.shape)
	#all_loc_loss = all_loc_loss * loss_weight_mask
	#print(all_loc_loss.shape)
	#mean_loc_loss = all_loc_loss.mean(dim=1)
	#print(mean_loc_loss.shape)
	#mean_loc_loss = mean_loc_loss.mean(dim=0)
	#print(mean_loc_loss)
	################################################################
	last_layer_points = vertex_coord_list[2].cpu().detach().numpy() if cuda else vertex_coord_list[2].detach().numpy()
	classification_last = predictions.cpu().detach().numpy() if cuda else predictions.detach().numpy()
	segmentation_last = segmentations.cpu().detach().numpy() if cuda else segmentations.detach().numpy()
	#############################################################
	
	
	###########################################################
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(111)
	im = ax.scatter(last_layer_points[:,0], last_layer_points[:,1],s=0.5,c=classification_last, cmap = "Reds")
	axes=matplotlib.pyplot.gca()
	axes.set_aspect(1)
	matplotlib.pyplot.savefig(postprocess_dir+"cls"+str(i)+".png", dpi=150)
	matplotlib.pyplot.close()
	matplotlib.pyplot.cla()
	matplotlib.pyplot.clf()
	
	cls_labels = cls_labels.cpu().detach().numpy() if cuda else cls_labels.detach().numpy()
	seg_labels =  seg_labels.cpu().detach().numpy() if cuda else seg_labels.detach().numpy()
	
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(111)
	im = ax.scatter(last_layer_v[:,0], last_layer_v[:,1],s=0.5,c=cls_labels,cmap = "Reds")
	axes=matplotlib.pyplot.gca()
	axes.set_aspect(1)
	matplotlib.pyplot.savefig(postprocess_dir+"cls"+str(i)+"_gt.png", dpi=150)
	matplotlib.pyplot.close()
	matplotlib.pyplot.cla()
	matplotlib.pyplot.clf()
	
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(111)
	im = ax.scatter(last_layer_points[:,0], last_layer_points[:,1],s=0.5,c=segmentation_last, cmap = "Reds")
	axes=matplotlib.pyplot.gca()
	axes.set_aspect(1)
	matplotlib.pyplot.savefig(postprocess_dir+"seg"+str(i)+".png", dpi=150)
	matplotlib.pyplot.close()
	matplotlib.pyplot.cla()
	matplotlib.pyplot.clf()
	
	#cls_labels = cls_labels.cpu().detach().numpy()
	
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(111)
	im = ax.scatter(last_layer_v[:,0], last_layer_v[:,1],s=0.5,c=seg_labels,cmap = "Reds")
	axes=matplotlib.pyplot.gca()
	axes.set_aspect(1)
	matplotlib.pyplot.savefig(postprocess_dir+"seg"+str(i)+"_gt.png", dpi=150)
	matplotlib.pyplot.close()
	matplotlib.pyplot.cla()
	matplotlib.pyplot.clf()

np.set_printoptions(threshold=False)
print(f"Done: {samples_count}/{samples_count}")
logfile.flush()
logfile.close()
	
