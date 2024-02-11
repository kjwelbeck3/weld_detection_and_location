import torch
from torch import nn
from Instseg_model import GraphNetAutoCenterToSingular, PointSetPooling, GraphNetAutoCenter


class LocationRegressionModelV1(nn.Module):

    def __init__(self, output_dims=3, graph_net_layers=1) -> None:
        super().__init__()
        self.output_dims = output_dims

        self.multi_point_pooling = PointSetPooling()
        
        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):
            self.graph_nets.append(GraphNetAutoCenter())

        self.single_point_feature_pooling = GraphNetAutoCenterToSingular(auto_offset=True, update_MLP_depth_list=[300, 64, 3])

   

    def forward(self, graph):

        # ### DEBUG TOKEN
        # print("graph.keys()")
        # print(graph.keys())

        vertex_coord_list = graph.get("vertex_coord_list")
        assert vertex_coord_list
        keypoint_indices_list = graph.get("keypoint_indices_list") 
        assert keypoint_indices_list
        edge_list = graph.get("edges_list")
        assert edge_list

        ## Point Coords to Point Feature Encoding
        point_coords = vertex_coord_list[0]
        keypoint_indices = keypoint_indices_list[0]
        pooling_edges = edge_list[0] 
        
        # ##### DEBUG TOKEN 
        # ## Checking the points, keypoint idx and edges used in the multi_point_pooling network
        # print("\nDEBUG")
        # print("point_coords")
        # print(point_coords.shape)
        # print("\nkeypoint_indices")
        # print(keypoint_indices.shape)
        # print("\npooling_edges")
        # print(pooling_edges.shape)

        point_features = self.multi_point_pooling(point_coords, keypoint_indices, pooling_edges)

        ## Pooling Point Features Through Intra graph edges then Updating
        point_coords = vertex_coord_list[1]
        keypoint_indices = keypoint_indices_list[0]
        pooling_edges = edge_list[1]
        for graph_net in self.graph_nets:
            point_features = graph_net(point_features, point_coords, keypoint_indices, pooling_edges)

        ## Pooling Point Features to Single Vertex then Mapping to Final Point Coords
        point_coords = vertex_coord_list[1]
        keypoint_indices = keypoint_indices_list[-1] # or explicitly [2]
        pooling_edges = edge_list[-1] # or explicitly [2]
        graph_coords = self.single_point_feature_pooling(point_features, point_coords, keypoint_indices, pooling_edges)

        return graph_coords

    def loss():
        pass

    def accuracy():
        pass