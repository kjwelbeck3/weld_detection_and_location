from datetime import datetime
from graph_generation import gen_multi_level_local_graph_v3
from dataset import pcloader
import numpy as np

## A logging utility
def log(logfile, message):
	stamp = datetime.now()
	str = f"[{stamp}] {message}"
	print(str)
	logfile.write(str)
	logfile.write("\n")

## Checking if the graph generation is determinisitc
## ie repeated call return the same downsampled set 

# def testing_graph_generation():
#     graph_gen_kwargs = {
#     'add_rnd3d': True,
#     'base_voxel_size': 0.8,
#     'downsample_method': 'random',
#     'level_configs': [
#         {'graph_gen_kwargs': {'num_neighbors': 64, 'radius': 0.4},
#          'graph_gen_method': 'disjointed_rnn_local_graph_v3',
#          'graph_level': 0,
#          'graph_scale': 1},
#         {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 1.2},
#          'graph_gen_method': 'disjointed_rnn_local_graph_v3',
#          'graph_level': 1,
#          'graph_scale': 1}]
#     }

#     scan_path = "./_data/scans/1.csv"
#     pcd_p, _ = pcloader(scan_path)

#     vertex_coord_listA, keypoint_indices_listA, edges_listA = \
#         gen_multi_level_local_graph_v3(pcd_p,0.6,graph_gen_kwargs['level_configs'])

#     vertex_coord_listB, keypoint_indices_listB, edges_listB = \
#         gen_multi_level_local_graph_v3(pcd_p,0.6,graph_gen_kwargs['level_configs'])

#     print("len(vertex_coord_listA)")
#     print(len(vertex_coord_listA))
#     print()
#     print("vertex_coord_listA[1]")
#     print(vertex_coord_listA[2])

#     print("vertex_coord_listB[1]")
#     print(vertex_coord_listB[2])
	

#     assert len(vertex_coord_listA) == len(vertex_coord_listB) , f'Vertex Coord Lists do not equate: {len(vertex_coord_listA)} != {len(vertex_coord_listB)}'
#     for i in range(min(len(vertex_coord_listA), len(vertex_coord_listB))):
#         assert np.array_equal(vertex_coord_listA[i], vertex_coord_listB[i]), f"Vertex Coord List[{i}] don't equate"
#     assert len(keypoint_indices_listA) == len(keypoint_indices_listB) , f'Keypoint Indices Lists do not equate: {len(keypoint_indices_listA)} != {len(keypoint_indices_listB)}'
#     assert len(edges_listA) == len(edges_listB) , f'Edge Lists do not equate: {len(edges_listA)} != {edges_listB}'

#     print("Check outs; Equating all around")

## Testing inferance on inverted .ply sample

def test_ply():
	from Instseg_model import MultiLayerFastLocalGraphModelV2 
	from postprocessor import process_sample, plot_sample
	print("Building Model...")
	model = MultiLayerFastLocalGraphModelV2(num_classes=3, max_instance_no=7)
	print("... Done")

	print("Loading Model parameters...")
	model_param_path = './_model/train1-fix1/2023_06_07_17_27_40/params_epoch49_2023_06_08_01_58_31.pt'
	print("... Done")

	# scan_path = './_data/test/3_RH-8-231201538-Pass-2023_06_07-12-06-46-655.ply'
	scan_path = './_data/test/4_RH-10-231201538-Pass-2023_06_07-12-06-55-935.ply'
	# label_path = './_data/labels/37.txt'
	label_path = None
	cuda = False

	print("Processing Sample ...")
	points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, label_path, model_param_path, cuda, cuda)
	# points, cls_preds, inst_preds, cls_labels, inst_labels = process_sample(scan_path, model, None, model_param_path, cuda, cuda)
	print("... Done")

	print("Plotting Sample ...")
	plot_sample(scan_path, points, "test1", "./results/test1/", cls_preds, inst_preds, cls_labels, inst_labels )
	print("... Done")

if __name__ == "__main__":

	# ## Testing the Logging Util
	# file = open("test_log.txt", 'w')
	# log(file, "something else")
	# log(file, {2: "he", 5: "hello"})
	# file.close()

	## Verifying determinism in graph generation
	# testing_graph_generation()
	## RESULT: NOT DETERMINISTIC; traced to the downsampling which introduces some randomization

	test_ply()
	
	