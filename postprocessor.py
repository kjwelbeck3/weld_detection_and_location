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
import csv

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


	# def process_new_scan(scan_path, model=None, model_param_path=None, cudatransfer_data=False, cudatransfer_model=False ):
	# 	a = time.time()

	# 	## tranfer: model to gpus
	# 	if cudatransfer_model and torch.cuda.is_available():
	# 		model = model.cuda()

	# 	if model_param_path:
	# 		model.load_state_dict(torch.load(model_param_path))

	# 	pointxyz, offset = pcloader(scan_path)
	# 	vertex_coord_list, keypoint_indices_list, edges_list = \
	# 	gen_multi_level_local_graph_v3(pointxyz,0.6,graph_gen_kwargs['level_configs'])
		
	# 	## conversions: type precision
	# 	vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
	# 	keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
	# 	edges_list = [e.astype(np.int32) for e in edges_list]

	# 	## conversions: numpy array to tensor
	# 	vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
	# 	keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
	# 	edges_list = [torch.from_numpy(item).long() for item in edges_list]
		
	# 	batch = (vertex_coord_list, keypoint_indices_list, edges_list)
	# 	new_batch = []

	# 	## tranfer: sample to gpus
	# 	if cudatransfer_data and torch.cuda.is_available():
	# 		for item in batch:
	# 			if not isinstance(item, torch.Tensor):
	# 					item = [x.cuda() for x in item]
	# 			else: 
	# 					item = item.cuda()
	# 			new_batch += [item]
	# 		vertex_coord_list, keypoint_indices_list, edges_list = new_batch
	# 		batch = new_batch

	# 	## inference
	# 	cls_seg, inst_seg = None, None
	# 	with torch.no_grad():
	# 		cls_seg, inst_seg = model(*batch)

	# 	cls_preds = torch.argmax(cls_seg, dim=1)
	# 	inst_preds = torch.argmax(inst_seg, dim=1)
	# 	b = time.time()
	# 	print("Time Elapsed (secs): ", b-a)

	# 	return cls_preds, inst_preds


def process_sample(scan_path, model, label_path=None, model_param_path=None, cudatransfer_data=False, cudatransfer_model=False ):
	a = time.time()

	## tranfer: model to gpus
	if cudatransfer_model and torch.cuda.is_available():
		model = model.cuda()

	if model_param_path:
		model.load_state_dict(torch.load(model_param_path))

	pointxyz, offset = pcloader(scan_path)
	vertex_coord_list, keypoint_indices_list, edges_list = \
	gen_multi_level_local_graph_v3(pointxyz,0.6,graph_gen_kwargs['level_configs'])
	last_layer_v = vertex_coord_list[-1]

	cls_labels, inst_labels = None, None
	if label_path:
		labellist = gtloader(label_path)
		Nv = last_layer_v.shape[0]
		cls_labels = np.zeros((Nv, 1), dtype=np.int64)
		inst_labels = np.zeros((Nv, 1), dtype=np.int64)
		
		instance_no = 0
		for label in labellist:
			instance_no +=1
			xcoor,ycoor,obj_cls = float(label[0]),float(label[1]),label[2]
			mask = assign_weld_region(last_layer_v,label,offset)
			cls_labels[mask, :] = obj_cls
			inst_labels[mask, :] = instance_no

	## conversions: type precision
	vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
	keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
	edges_list = [e.astype(np.int32) for e in edges_list]

	## conversions: numpy array to tensor
	vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
	keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
	edges_list = [torch.from_numpy(item).long() for item in edges_list]

	if label_path:
		cls_labels = cls_labels.astype(np.int32)
		inst_labels = inst_labels.astype(np.float32)
		cls_labels = torch.from_numpy(cls_labels)
		inst_labels = torch.from_numpy(inst_labels)
	
	batch = (vertex_coord_list, keypoint_indices_list, edges_list)
	new_batch = []

	## tranfer: sample to gpus
	## does not send labels to gpus; inference only; no training
	if cudatransfer_data and torch.cuda.is_available():
		for item in batch:
			if not isinstance(item, torch.Tensor):
					item = [x.cuda() for x in item]
			else: 
					item = item.cuda()
			new_batch += [item]
		vertex_coord_list, keypoint_indices_list, edges_list = new_batch
		batch = new_batch

	## inference
	cls_seg, inst_seg = None, None
	with torch.no_grad():
		cls_seg, inst_seg = model(*batch)

	cls_preds = torch.argmax(cls_seg, dim=1)
	inst_preds = torch.argmax(inst_seg, dim=1)
	b = time.time()
	print("Time Elapsed (secs): ", b-a)

	if cudatransfer_data and torch.cuda.is_available():
		cls_preds = cls_preds.cpu().detach().numpy()
		inst_preds= inst_preds.cpu().detach().numpy()

	# print("last_layer_v.shape")
	# print(last_layer_v.shape)
	
	# print("cls_preds.shape")
	# print(cls_preds.shape)
	
	# print("inst_preds.shape")
	# print(inst_preds.shape)
	
	# print("cls_labels.shape")
	# print(cls_labels.shape)
	
	# print("inst_labels.shape")
	# print(inst_labels.shape)

	return last_layer_v, cls_preds, inst_preds, cls_labels, inst_labels


def plot_sample(scan_path, points, tag="", output_dir="./results/plots/", cls_preds=None, inst_preds=None, cls_labels=None, inst_labels=None):

	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	starttime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
	# cls_accuracy, inst_accuracy = None, None

	if cls_labels is not None:
		title = "Class Segmentation Labels"
		captions = f"Scan: {scan_path}"
		path = f"{output_dir}{tag}_cls_{starttime}_gt.png"
		_plot(path, points, cls_labels, cmap="Reds", title=title, caption=captions)

	if inst_labels is not None:
		title = "Instance Segmentation Labels"
		captions = f"Scan: {scan_path}"
		path = f"{output_dir}{tag}_inst_{starttime}_gt.png"
		_plot(path, points, inst_labels, cmap="Reds", caption=captions, title=title)

	if cls_preds is not None:
		title = "Class Segmentation"
		captions = f"Scan: {scan_path}"
		if cls_labels is not None:
			acc = compute_accuracy(cls_preds, cls_labels)
			captions += f"\nAccuracy: {acc}"
		
		path = f"{output_dir}{tag}_cls_{starttime}.png"
		_plot(path, points, cls_preds, cmap="Reds", caption=captions, title=title)

	if inst_preds is not None:
		captions = f"Scan: {scan_path}"
		title = "Instance Segmentation"
		if inst_labels is not None:
			acc = compute_accuracy(inst_preds, inst_labels)
			captions += f"\nAccuracy: {acc}"

		path = f"{output_dir}{tag}_inst_{starttime}.png"
		_plot(path, points, inst_preds, cmap="Reds", caption=captions, title=title)

	
def _plot(path, points, values, cmap, title="", caption=""):
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(111)
	ax.set_title(title)
	im = ax.scatter(points[:,0], points[:,1],s=0.5,c=values,cmap = cmap)
	
	ax.set_xlabel("x [mm]")
	ax.set_ylabel("y [mm]")
	legend_ = ax.legend(*im.legend_elements(), bbox_to_anchor=(1.1, 1), loc="upper right")
	# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
	ax.add_artist(legend_)
	ax.text(0.5, -0.25, caption, style='italic', \
		horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
	axes=matplotlib.pyplot.gca()
	axes.set_aspect(1)
	matplotlib.pyplot.savefig(path, dpi=150)
	matplotlib.pyplot.close()
	matplotlib.pyplot.cla()
	matplotlib.pyplot.clf()

def compute_accuracy(preds, labels):
	preds = preds.squeeze()
	labels = labels.squeeze()
	assert preds.shape == labels.shape, "Predictions and Labels must match in size"

	true_count = int((preds == labels).sum())
	total_count = preds.shape[0]

	return true_count/total_count



## CORRECTION FOR LATER: Cudatransfer_model and _data should really be the same thing
def process_and_plot_list(sample_list, model, model_param_path=None, cudatransfer_data=False, 
						cudatransfer_model=False, tag="", output_dir="./results/plots/", overwrite_dir=True, logfile=None):

	if overwrite_dir:
		if os.path.isdir(output_dir):
			import shutil
			shutil.rmtree(output_dir)
		os.makedirs(output_dir)

	starttime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

	## tranfer: model to gpus
	if cudatransfer_model and torch.cuda.is_available():
		model = model.cuda()

	if model_param_path:
		model.load_state_dict(torch.load(model_param_path))

	np.set_printoptions(threshold=sys.maxsize)
	torch.set_printoptions(threshold=sys.maxsize)

	points, cls_preds, inst_preds, cls_labels, inst_labels = None, None, None, None, None
	list_count = len(sample_list)

	print(f"{output_dir}{tag}_{starttime}.csv")
	with open(f"{output_dir}{tag}_{starttime}.csv", mode='w') as csv_file:
		
		csv_writer = csv.writer(csv_file, delimiter=',')
		csv_writer.writerow(["idx", "cls_accuracy", "inst_accuracy", "scan_path"])
		for idx, item in enumerate(sample_list, 1):
			print(f"Processing: {idx}/{list_count}")
			scan_path, label_path = None, None
			if len(item) == 2:
				scan_path, label_path = item
			elif len(item) == 1:
				scan_path = item[0]
			points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, 
																				label_path=label_path, 
																				model_param_path=None, 
																				cudatransfer_data=cudatransfer_data, 
																				cudatransfer_model=cudatransfer_model)

			cls_accuracy = compute_accuracy(cls_preds, cls_labels)
			inst_accuracy = compute_accuracy(inst_preds, inst_labels)
			csv_writer.writerow([idx, cls_accuracy, inst_accuracy, scan_path])

			if logfile:
				logfile.write(f"{idx}/{list_count}")
				logfile.write('points:\n')
				logfile.write(f"{points}")
				logfile.write('cls_preds:\n')
				logfile.write(f"{cls_preds}")
				logfile.write('inst_preds:\n')
				logfile.write(f"{inst_preds}")
				logfile.write('cls_labels:\n')
				logfile.write(f"{cls_labels}")
				logfile.write('inst_labels:\n')
				logfile.write(f"{inst_labels}")
				logfile.write("####\n\n")
				logfile.flush()

			plot_sample(scan_path, points, 
						tag=f"{tag}{idx}", output_dir=output_dir, 
						cls_preds=cls_preds, inst_preds=inst_preds, 
						cls_labels=cls_labels, inst_labels=inst_labels)
		
	np.set_printoptions(threshold=False)

def process_and_plot_from_text(file_path, model, subrange=(0,None), model_param_path=None, cudatransfer_data=False, 
						cudatransfer_model=False, tag="", output_dir="./results/plots/"):

	if not os.path.isdir(output_dir):
		print(f"Could not find output_dir, {output_dir}")
		print("Creating output_dir...")
		os.makedirs(output_dir)
		print("... created")

	data_list = []

	with open(file_path, "r") as textfile:
		samples = textfile.readlines()
		for sample in samples:
			# train_set.append(eval(sample))
			_sample = eval(sample)
			data_list.append(_sample)
	data_list = data_list[subrange[0]:subrange[1]]

	## Check that teach of these locations exit
	scan_exists_count = 0
	samples_count = 0
	label_exists_count = 0

	for scan, label in data_list:
		samples_count+=1    
		if os.path.isfile(scan):
			scan_exists_count+=1
		else:
			print(f"Does not exist: {scan}")
		if os.path.isfile(label):
			label_exists_count+=1
		else:
			print(f"Does not exist: {label}")

	print(f"samples_count: {samples_count}")
	print(f"scan_exists_count: {scan_exists_count}")
	print(f"label_exists_count: {scan_exists_count}")

	logfile = open(f"{output_dir}print_statements.txt", 'w')

	process_and_plot_list(data_list, model, model_param_path=model_param_path, cudatransfer_data=cuda, overwrite_dir=False, 
						cudatransfer_model=cuda, tag=tag, output_dir=output_dir, logfile=logfile)

	logfile.close()


## CORRECTION FOR LATER: Cudatransfer_model and _data should really be the same thing
def process_list(sample_list, model, model_param_path=None, cudatransfer_data=False, 
						cudatransfer_model=False, tag="", output_dir="./results/plots/", overwrite_dir=False, logfile=None):

	starttime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

	## tranfer: model to gpus
	if cudatransfer_model and torch.cuda.is_available():
		model = model.cuda()

	if model_param_path:
		model.load_state_dict(torch.load(model_param_path))

	np.set_printoptions(threshold=sys.maxsize)
	torch.set_printoptions(threshold=sys.maxsize)

	points, cls_preds, inst_preds, cls_labels, inst_labels = None, None, None, None, None
	list_count = len(sample_list)

	print(f"{output_dir}{tag}_{starttime}.csv")
	with open(f"{output_dir}{tag}_{starttime}.csv", mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',')
		csv_writer.writerow(["idx", "cls_accuracy", "inst_accuracy", "scan_path"])
		for idx, item in enumerate(sample_list, 1):
			print(f"Processing: {idx}/{list_count}")
			scan_path, label_path = None, None
			if len(item) == 2:
				scan_path, label_path = item
			elif len(item) == 1:
				scan_path = item[0]
			points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, 
																				label_path=label_path, 
																				model_param_path=None, 
																				cudatransfer_data=cudatransfer_data, 
																				cudatransfer_model=cudatransfer_model)

			cls_accuracy = compute_accuracy(cls_preds, cls_labels)
			inst_accuracy = compute_accuracy(inst_preds, inst_labels)
			csv_writer.writerow([idx, cls_accuracy, inst_accuracy, scan_path])

			if logfile:
				logfile.write(f"{idx}/{list_count}")
				logfile.write('points:\n')
				logfile.write(f"{points}")
				logfile.write('cls_preds:\n')
				logfile.write(f"{cls_preds}")
				logfile.write('inst_preds:\n')
				logfile.write(f"{inst_preds}")
				logfile.write('cls_labels:\n')
				logfile.write(f"{cls_labels}")
				logfile.write('inst_labels:\n')
				logfile.write(f"{inst_labels}")
				logfile.write("####\n\n")
				logfile.flush()
		
	np.set_printoptions(threshold=False)
	print("Done: Processing List")

def process_from_text(file_path, model, subrange=(0,None), model_param_path=None, cudatransfer_data=False, 
						cudatransfer_model=False, tag="", output_dir="./results/plots/"):

	if not os.path.isdir(output_dir):
		print(f"Could not find output_dir, {output_dir}")
		print("Creating output_dir...")
		os.makedirs(output_dir)
		print("... created")


	data_list = []

	with open(file_path, "r") as textfile:
		samples = textfile.readlines()
		for sample in samples:
			# train_set.append(eval(sample))
			_sample = eval(sample)
			data_list.append(_sample)
	data_list = data_list[subrange[0]:subrange[1]]

	## Check that teach of these locations exit
	scan_exists_count = 0
	samples_count = 0
	label_exists_count = 0

	for scan, label in data_list:
		samples_count+=1    
		if os.path.isfile(scan):
			scan_exists_count+=1
		else:
			print(f"Does not exist: {scan}")
		if os.path.isfile(label):
			label_exists_count+=1
		else:
			print(f"Does not exist: {label}")

	print(f"samples_count: {samples_count}")
	print(f"scan_exists_count: {scan_exists_count}")
	print(f"label_exists_count: {scan_exists_count}")

	logfile = open(f"{output_dir}print_statements.txt", 'w')

	process_list(data_list, model, model_param_path=model_param_path, cudatransfer_data=cuda, 
						cudatransfer_model=cuda, tag=tag, output_dir=output_dir, logfile=logfile)

	logfile.close()
	print(f"Done: Processing file {file_path}")


if __name__ == "__main__":

	print("Building Model...")
	model = MultiLayerFastLocalGraphModelV2(num_classes=3, max_instance_no=7)
	print("... Done")

	print("Loading Model parameters...")
	# model_param_path = './_model/train1-fix1/2023_06_07_17_27_40/params_epoch49_2023_06_08_01_58_31.pt'
	# model_param_path = './_model/train1-fix2/params_epoch49_for_min_test_loss.pt'
	# model_param_path = './_model/train1-fix2/2023_06_15_09_03_24/params_epoch97_for_min_test_loss.pt'
	model_param_path = './_model/train1-fix3/2023_06_30_09_56_12/params_epoch488_for_min_test_loss.pt'
	print("... Done")
	
	scan_path = './_data/scans/37.csv'
	label_path = './_data/labels/37.txt'
	sample_list = [[scan_path], [scan_path], [scan_path, label_path] ]
	sample_textfile = './_data/.txt'
	sample_range = (0, 10)
	cuda = False

	# ## Testing a Single Sample
	# print("Processing Sample ...")
	# points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, label_path, model_param_path, cuda, cuda)
	# # points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, None, model_param_path, cuda, cuda)
	# print("... Done")

	# print("Plotting Sample ...")
	# plot_sample(scan_path, points, "test1", "./results/test1/", None, inst_preds, None, inst_labels )
	# print("... Done")


	# ## Testing a Sample List
	# process_and_plot_list(sample_list, model, model_param_path=model_param_path, cudatransfer_data=cuda, 
	# 					cudatransfer_model=cuda, tag="temp", output_dir="./results/test1/")
	# print("... Done")


	# ## Testing a Sample Text File
	# process_and_plot_from_text(sample_textfile, model, subrange=sample_range, 
	# 					model_param_path=model_param_path, tag="temp_by_file",
	# 					output_dir="./results/test2/")
	# print("... Done")

	## Testing a Sample Text File
	sample_textfile = './_data/selection.txt'
	process_and_plot_from_text(sample_textfile, model,  subrange=(0, None),
						model_param_path=model_param_path, tag="selection_e1000",
						output_dir="./results/train1-fix3-epochs1000/selection/")
	print("... Done")

