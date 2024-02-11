from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.utils.data as data_utils
import os
from graph_generation import gen_multi_level_local_graph_v3

locationing_graph_gen_kwargs_v1 = {
    'add_rnd3d': False,
    'base_voxel_size': 2,
    'level_configs': [

        {'graph_gen_kwargs': {'num_neighbors': 64, 'radius': 1},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 0,
         'graph_scale': 1},
        {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 5},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 1,
         'graph_scale': 1},
        #  {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 5},
        #  'graph_gen_method': 'disjointed_rnn_local_graph_v3',
        #  'graph_level': 2,
        #  'graph_scale': 2}

         ]
}

def print_graph_sizes(graph):
    for key in graph.keys():
        print(f"\n{key}")
        if isinstance(graph[key], list):
            for i,el in enumerate(graph[key]):
                if isinstance(el, np.ndarray):
                    print("--{i}:", el.shape)
        elif isinstance(graph[key], np.ndarray):
           print("-", graph[key]) 
    print()

def pcloader(file_path):
    if file_path.endswith(".npy"):
        return np.load(file_path)
    
    return None
    

def gtloader(file_path):
    with open(file_path, 'r') as f:
        label_text = f.readline()
        label = [float(el) for el in label_text.split()]
    return label

class MyLocationingDataSet(data_utils.Dataset):
    def __init__(self, pc_dir, label_dir, unit_scaling=True, pc_loader=pcloader, label_loader=gtloader):
        
        assert os.path.isdir(pc_dir), "Misspecified pc_dir."
        self.pc_dir = pc_dir
        
        assert os.path.isdir(label_dir), "Misspecified label_dir."
        self.label_dir = label_dir 

        pc_tags = [filename.split(".")[0] for filename in os.listdir(self.pc_dir)]
        label_tags = [filename.split(".")[0] for filename in os.listdir(self.label_dir)]

        self.dataset = [(tag+".npy", tag+".txt") for tag in pc_tags if tag in label_tags]

        ## Reporting Summary
        print(f"Found {len(pc_tags)} point cloud files")
        print(f"Found {len(label_tags)} label files")
        print(f"#matches = {len(self.dataset)}")
        print(f"#mismatches = {len(pc_tags)-len(self.dataset)} + {len(label_tags)-len(self.dataset)}")

        
        # ## DEBUG TOKEN
        # print("dataset members:")
        # print(self.dataset)

        self.scaling = unit_scaling
        self.pc_loader = pc_loader
        self.label_loader = label_loader

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        pc_path, label_path = self.dataset[index]
        pc_path = self.pc_dir+pc_path
        label_path = self.label_dir+label_path
        
        # ## DEBUG TOKEN
        # print("pc_path:")
        # print(pc_path)
        # print("exists?:", os.path.isfile(pc_path), "\n")
        # print("label_path:")
        # print(label_path)
        # print("exists?:", os.path.isfile(label_path), "\n")

        pc = self.pc_loader(pc_path)
        label = self.label_loader(label_path)
        label = np.array(label).reshape((1,3)).astype(np.float32)

        if self.scaling:
            self.pc_min = np.amin(pc[:, :2],0)
            self.pc_max = np.amax(pc[:, :2],0)
            pc_span = self.pc_max - self.pc_min
            pc  = (pc[:, :2] - self.pc_min)/self.pc_span

            label = (label[:, :2] - self.pc_min)/self.pc_span
            
            
             
        
        # ## DEBUG TOKEN
        # print("pc:")
        # print(pc.shape)

        # print("\nlabel")
        # print(label)

        ########################
        ## CONSTRUCTING MULTI-LEVEL GRAPH FROM POINT CLOUD
        ########################

        ## Generate multi(2)-layer graph from weld points
        ## settings:
        base_voxel_size = locationing_graph_gen_kwargs_v1['base_voxel_size']
        if self.scaling:
            base_voxel_size = base_voxel_size*np.max(self.pc_span)
        level_configs = locationing_graph_gen_kwargs_v1['level_configs']
        add_rnd3d = locationing_graph_gen_kwargs_v1['add_rnd3d']
        vertex_coord_list, keypoint_indices_list, edges_list = \
        gen_multi_level_local_graph_v3( \
                pc, \
                base_voxel_size, \
                level_configs, \
                add_rnd3d=add_rnd3d\
                    )

        ## Adding the last layer of single keypoint
        ## Possible options:
        ## - mean
        ## - random choice from previous layer
        ## - median/mode/range-mid
        ## - any above plus NN-based offset

        ## Option 1: mean
        final_point = np.expand_dims(np.mean(vertex_coord_list[-1], axis=0), axis=0)
        c = vertex_coord_list[-1].shape[0]
        final_edges = np.hstack((np.arange(c).reshape((c,1)), np.zeros((c, 1))))
        final_keypoint_indices = np.expand_dims(np.array([0]), axis=0)

        vertex_coord_list.append(final_point)
        keypoint_indices_list.append(final_keypoint_indices)
        edges_list.append(final_edges)

        ## Type casting
        ##  - Graph Components to floats and ints(for indices)
        vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
        keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
        edges_list = [e.astype(np.int32) for e in edges_list]
        
        ## HANDLED ABOVE
        # ##  - Label to floats and reshaped 
        # label = np.array(label).reshape((1,3)).astype(np.float32)

        ## Converting from numpy data types to torch equivalent
        ##  - Graph Components
        vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
        keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
        edges_list = [torch.from_numpy(item).long() for item in edges_list]
        ##  - Label
        label = torch.from_numpy(label)

        ## Packaging Outputs into a dict
        graph = { "vertex_coord_list": vertex_coord_list, \
                    "keypoint_indices_list": keypoint_indices_list, \
                    "edges_list" : edges_list }

        # ### DEBUG TOKEN -- DIMENSION CHECK
        # print_graph_sizes(graph)

        return graph, label