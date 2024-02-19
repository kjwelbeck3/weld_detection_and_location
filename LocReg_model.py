import torch
from torch import nn
# from torch.nn.functional import sigmoid, MSE
from Instseg_model import GraphNetAutoCenterToSingular, PointSetPooling, GraphNetAutoCenter


class LocationRegressionModelV1(nn.Module):

    def __init__(self, output_dims=3, graph_net_layers=1, scaling="sigmoid") -> None:
        super().__init__()
        self.output_dims = output_dims

        self.multi_point_pooling = PointSetPooling()
        
        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):
            self.graph_nets.append(GraphNetAutoCenter())

        self.single_point_feature_pooling = GraphNetAutoCenterToSingular(auto_offset=True, update_MLP_depth_list=[300, 64, 3])

        self.scaling = scaling
        # if self.scaling == "sigmoid":
            # self.


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
        # centroid_coords = torch.mean(point_coords, dim=0, keepdims=True)

        
        
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

        # pool_coords = vertex_coord_list[-1]

        return graph_coords

    # def postprocess()

    # def loss(self, pred_loc, label_loc, centroid_loc, pool_loc):
    #     ## Already rezeroed when loaded as a sample

    #     ## GOAL: 
    #     # Calc sigmoid on the prediction and on the label/centroid/pool point
    #     # Use outputs in MSE losses, with respective weighting

    #     pred_loc_sig = nn.functional.sigmoid(pred_loc)
    #     label_loc_sig = nn.functional.sigmoid(label_loc)
    #     centroid_loc_sig = nn.functional.sigmoid(centroid_loc)
    #     pool_loc_sig = nn.functional.sigmoid(pool_loc)
        
    #     params = torch.cat([x.view(-1) for x in self.parameters()])
    #     reg_loss = torch.mean(params.abs())
        
    #     loss_dict = {}
    #     loss_dict.update({"pred"})

        



        

        pass

    def accuracy():
        pass

class LogReg_MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prediction, label, centroid, pool_pt, 
        coeff_label=1, coeff_centroid=1, coeff_pool_pt=1):

        label_loss = torch.mean((prediction - label)**2)
        centroid_loss = torch.mean((prediction - centroid)**2)
        pool_pt_loss = torch.mean((prediction - pool_pt)**2)
        
        ## combined: weighted sum
        combined = label_loss*coeff_label + centroid_loss*coeff_centroid + pool_pt_loss*coeff_pool_pt

        # ## nominal loss: ie distance to label
        # nominal = torch.sum((prediction - label)**2)**(0.5)

        return combined#, nominal
        

class LogReg_MSELoss_v2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prediction, target):

        ## Same Reference frame eg. zeroed at pool_point, or pc_min
        mse = torch.mean((prediction - target)**2)

        # ## nominal loss: ie distance to target
        # nominal_dist = torch.sum((prediction - target)**2)**(0.5)
        

        return mse#, nominal_dist