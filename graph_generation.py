import numpy as np
from sklearn.neighbors import NearestNeighbors

## Sample Graph Definition to Transform Point Cloud to Graph
## For constructing layers of points, keypoints, and the inter- and intra-layer edges
graph_gen_kwargs = {
    'add_rnd3d': True,
    'base_voxel_size': 0.8,
    'downsample_method': 'random',
    'level_configs': [
        {'graph_gen_kwargs': {'num_neighbors': -1, 'radius': 0.2},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 0,
         'graph_scale': 1},
        {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 0.8},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 1,
         'graph_scale': 1}]
}


def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales by voxelizing then randomly selecting a point
    within a voxel cell.
    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates
        base_voxel_size: scalar, the cell size of voxel.
        levels: a list of successive downsampling scales.
            eg. [1] downsamples the original points by base_voxel_size creating a new and final layer
            eg. [1, 1] downsample orig points for new layer, then duplicates for another new and final layer
        add_rnd3d: boolean, whether to add random offset when downsampling.
   
    returns: vertex_coord_list, keypoint_indices_list

    Downsamples (relative to previous level of the graph) point by a specified resolutions on voxel size
    Saving the xyz locations and the index of the point in respective lists
    For each level of downsampling, 
        function extends both the vertex_coord_list and the keypoint_indices_list with a new array.
    """
    ## Adding missing z dimension if missing, setting to same zero-plane
    c, d =  points_xyz.shape
    if d == 2:
        points_xyz = np.hstack((points_xyz, np.zeros((c, 1)))) 

    # xyz_offset would zero all points w.r.t the min for each axis
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)   ##!!! EXCESSIVE/UNNECESSARY/UNUSED
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])

    # ########## DEBUG TOKEN
    # print("xyz_offset")
    # print(xyz_offset)

    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)   ##!!! EXCESSIVE/UNNECESSARY/UNUSED
    vertex_coord_list = [points_xyz]
    keypoint_indices_list = []
    last_level = 0
    for level_id, level in enumerate(levels):
        last_points_xyz = vertex_coord_list[-1]
        if np.isclose(last_level, level):
            
            # same downsample scale (gnn layer), just copy it
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            ## Bucketting each point into respective the x, y and z bins;
            ## Resolution of bins set by the base voxel size 
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                    // (base_voxel_size*level)
            else:
                ## add_rnd3d additionally adds an offset which may move some points to adjacaent buckets in any direction
                xyz_idx = (last_points_xyz - xyz_offset +
                    base_voxel_size*level*np.random.random((1,3))) \
                        // (base_voxel_size*level)
                
            # ########## DEBUG TOKEN
            # print("xyz_idx")
            # print(xyz_idx)

            xyz_idx = xyz_idx.astype(np.int32)  ## happens to remove any offsets added, thus undoing the if else block above 
                                                ## for the offsets that did not skip points into adjacent buckets
            
            # ########## DEBUG TOKEN
            # print("xyz_idx as int")
            # print(xyz_idx)
            
            
            dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1  ## counting the number of divisions/buckets

            #Keys for each point pointing to  3d bucket location = x_loc + y_loc*x_size + z_loc*x_size*y_size 
            keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
            num_points = xyz_idx.shape[0]

            ## for each point in the last points, find the point's bucket location
            ## build up list of the indexes of points located at each bucket location
            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            # ########## DEBUG TOKEN
            # print("level_id", level_id)
            # print("voxels_idx")
            # print(voxels_idx)
            
            ## Random selection of point within each bucket location
            ## Saving the point's location, and its index in the list of points
            downsampled_xyz = []
            downsampled_xyz_idx = []
            for key in voxels_idx:
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)

            ## Add the downsampled points as next graph level to to vertex_coord list
            ## Add downsampling index to keypoint_indices_list
            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
        last_level = level

    return vertex_coord_list, keypoint_indices_list

def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
    """Generate a local graph by radius neighbors.
        Args: 
            points_xyz: [Nx3] Matrix representing upstream layer of points
            center_xyz: [Kx3] Matrix representing downstream layer points
            radius: scalar dist within which nearest neighbor utility classifies as neighbors
            num_neighbors: scalar, max count of upstream layer neighbor for each downstream layer point
            scale: scaling factor on the point cloud
        Output:
            vertices: [Ex2] ==> source_idx-to-destination_idx

    """
    if scale is not None:
        scale = np.array(scale)
        points_xyz = points_xyz/scale
        center_xyz = center_xyz/scale

    ## Train Nearest Neighbors on the upstream collection of points
    ## Query downstream collection to return index list of training/source points for each test point
    ## altogether in a list of lists  
    nbrs = NearestNeighbors(
        radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)

    # ##### DEBUG  TOKEN
    # print("indices")
    # print(indices)


    ## Cap number of listed source neighbor points to num_neighbors 
    ## Randomize if downselecting
    if num_neighbors > 0:
        if neighbors_downsample_method == 'random':
            indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]

    ## The list of list source points enumerated in a single list
    ## Corresponding list of destinations points to match each pairing
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices
    ## output: [#edges by 2] ==> source_idx-to-destination_idx by  

def gen_multi_level_local_graph_v3(
    points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
    downsample_method='random'):
    """Generating graphs at multiple scales. This function enforces that output
    vertices of a graph layer matches the input vertices of next graph layer so that
    gnn layers can be applied sequentially.
    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a list of dicts of 'level', 'graph_gen_method','graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
        downsample_method: string, the name of downsampling method.
    returns: 
        vertex_coord_list: list of ?x3 Matrices representing the multiple layers of the graph 
        keypoint_indices_list: list of lists of successive keypoint extractions from upstream layers post-voxelizing 
        edges_list: list of ?x2 matrix representing the upstream_source_idx-to-downstream_source_idx edge mappings
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)

    # Gather the downsample scale for each graph
    scales = [config['graph_scale'] for config in level_configs]

    # Generate vertex coordinates per multi-layer
    if downsample_method=='random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    
    # Create edges
    edges_list = []
    for config in level_configs:
        graph_level = config['graph_level']
        method_kwarg = config['graph_gen_kwargs']
        points_xyz = vertex_coord_list[graph_level]  #edge source
        center_xyz = vertex_coord_list[graph_level+1] #edge destination
        vertices = gen_disjointed_rnn_local_graph_v3(points_xyz, center_xyz, **method_kwarg)
        edges_list.append(vertices)
    return vertex_coord_list, keypoint_indices_list, edges_list


