import os
import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader
import numpy as np
from LocReg_model import LocationRegressionModelV1, LogReg_MSELoss, LogReg_MSELoss_v2
from locationing_dataset import MyLocationingDataSet as LocDataset
from train1 import log, plot_inference_loc
from datetime import datetime

## Runtime Settings
cuda = True
np.random.seed(0)   
# device = (
#         "cuda"
#         if torch.cuda.is_available() and cuda
#         else "cpu"
#     )
device="cpu"


## Data Source Settings
train_root_dir = "C:/Users/KZTYLF/Documents/playground/GNN UIs/GNN InstanceSegmentation/Recreating Dataset/locationing_dataset/train/"
test_root_dir = "C:/Users/KZTYLF/Documents/playground/GNN UIs/GNN InstanceSegmentation/Recreating Dataset/locationing_dataset/test/"

## Model Params Paths
model_params_test_dir = "C:/Users/KZTYLF/Documents/playground/GNN UIs/GNN InstanceSegmentation/Recreating Dataset/locationing_dataset/trash/"
model_params_tag = "params_"

def get_filename(f_dir, f_tag, format_suffix=".pth" ):
    """
    Counts up and Returns the next file name with the same tag in the spec'd dir
    """
    tag_matches = [f for f in os.listdir(f_dir) if f.startswith(f_tag)]
    return f"{f_dir}{f_tag}_{len(tag_matches)}{format_suffix}"

#### **Packaged as function for use in test py notebooks
def setupDataLoaders(train_root_dir, test_root_dir=None, batch_size=1, logfile=None, scaling=None):
    
    ## [TO FIX]
    # On account of different point cloud sizes, cannot batch (ie size>1) load using available utilities
    # As such the dataloader is not necessary and should be removed from implementation 


    train_pc_dir = train_root_dir+"point_clouds/"
    train_label_dir = train_root_dir+"labels/"
    
    if test_root_dir:
        test_pc_dir = test_root_dir+"point_clouds/"
        test_label_dir = test_root_dir+"labels/"

    ## Preparing the training and test data
    training_dataset = LocDataset(train_pc_dir, train_label_dir, scaling=scaling)
    if test_root_dir:
        test_dataset = LocDataset(test_pc_dir, test_label_dir, scaling=scaling)

    ##  - DataLoaders
    train_dataloader = DataLoader(training_dataset, batch_size=1)


    # if test_root_dir:
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for X, y in test_dataloader:
        # print(f"Shape of X (test) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
        # print(f"Shape of y (test): {y.shape} {y.dtype}")
        # print(f"Dataset Size: {len(test_dataloader.dataset)}")
        # print(f"DataLoader Batch Size: {len(test_dataloader)}")
        # break

        log(logfile, message=f"Sample from Test Set")
        log(logfile, message=f"Shape of X (test) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
        log(logfile, message=f"Shape of y (test): {y.shape} {y.dtype}")
        log(logfile, message=f"Dataset Size: {len(test_dataloader.dataset)}")
        log(logfile, message=f"DataLoader Batch Size: {len(test_dataloader)}")
        break

    for X, y in train_dataloader:
        # print(f"Shape of X (train) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
        # print(f"Shape of y (train): {y.shape} {y.dtype}")
        # print(f"Dataset Size: {len(train_dataloader.dataset)}")
        # print(f"DataLoader Batch Size: {len(train_dataloader)}")
        # break

        log(logfile, message=f"Sample from Training Set")
        log(logfile, message=f"Shape of X (train) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
        log(logfile, message=f"Shape of y (train): {y.shape} {y.dtype}")
        log(logfile, message=f"Dataset Size: {len(train_dataloader.dataset)}")
        log(logfile, message=f"DataLoader Batch Size: {len(train_dataloader)}")
        break

    return train_dataloader, test_dataloader

def setupDataLoaders_v2(train_root_dir, test_root_dir=None, batch_size=1, logfile=None):
    """
    For working mini-batching using the collate_fn arg of DataLoader construction
    """
    ## [TO FIX]
    # On account of different point cloud sizes, cannot batch (ie size>1) load using available utilities
    # As such the dataloader is not necessary and should be removed from implementation 


    train_pc_dir = train_root_dir+"point_clouds/"
    train_label_dir = train_root_dir+"labels/"
    
    if test_root_dir:
        test_pc_dir = test_root_dir+"point_clouds/"
        test_label_dir = test_root_dir+"labels/"

    ## Preparing the training and test data
    training_dataset = LocDataset(train_pc_dir, train_label_dir)
    if test_root_dir:
        test_dataset = LocDataset(test_pc_dir, test_label_dir)

    ##  - DataLoaders
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=collate_labelled_graphs)

    if test_root_dir:
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        for X, y in test_dataloader:
            # print(f"Shape of X (test) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
            # print(f"Shape of y (test): {y.shape} {y.dtype}")
            # print(f"Dataset Size: {len(test_dataloader.dataset)}")
            # print(f"DataLoader Batch Size: {len(test_dataloader)}")
            # break

            log(logfile, message=f"Sample from Test Set")
            log(logfile, message=f"Shape of X (test) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
            log(logfile, message=f"Shape of y (test): {y.shape} {y.dtype}")
            log(logfile, message=f"Dataset Size: {len(test_dataloader.dataset)}")
            log(logfile, message=f"DataLoader Batch Size: {len(test_dataloader)}")
            break

    for X, y in train_dataloader:
        # print(f"Shape of X (train) [N, C, H, W]: {len(X)}, {X['vertex_coord_list'][-1].shape}")
        # print(f"Shape of y (train): {y.shape} {y.dtype}")
        # print(f"Dataset Size: {len(train_dataloader.dataset)}")
        # print(f"DataLoader Batch Size: {len(train_dataloader)}")
        # break

        log(logfile, message=f"Sample from Training Set")
        log(logfile, message=f"Shape of X (train) [N, C, H, W]: {len(X)}, {[X[el][0] for el in X.keys()] }")
        # log(logfile, message=f"Shape of X (train) [N, C, H, W]: {len(X)}, {X.keys()}['vertex_coord_list'][-1].shape}")
        # log(logfile, message=f"Shape of y (train): {len(y)}, {y[0].shape}")
        log(logfile, message=f"Shape of y (train): {y.shape}, {y.dtype}")
        log(logfile, message=f"Dataset Size: {len(train_dataloader.dataset)}")
        log(logfile, message=f"DataLoader Batch Size: {len(train_dataloader)}")
        break

    return train_dataloader, test_dataloader


#### **Packaged as function for use in test py notebooks
def setupLogRegModel(model_params_path=None, logfile=None):
    # print(f"Using {device} device")
    log(logfile, message=f"Using {device} device")
    model = LocationRegressionModelV1().to(device)
    # print(f"Model: Instantiated to {device}")
    log(logfile, message=f"Model: Instantiated to {device}")

    if model_params_path:
        model.load_state_dict(torch.load(model_params_path))   
        # print(f"Model: Parameters loaded from {model_params_path}")
        log(logfile, message=f"Model: Parameters loaded from {model_params_path}")

    return model

#### **Packaged as function for use in test py notebooks
# def setupLocRegOptimizer(model, learning_rate=1e-3, reg_constant=0, logfile=None):
#     loss_fn = nn.MSELoss()

#     # print("Optimizer: MSE Loss Function")
#     log(logfile, message="Optimizer: MSE Loss Function")

#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=reg_constant)
#     # print(f"Optimizer: SDG with learning rate={learning_rate} and regularization constant={reg_constant}")
#     log(logfile, message=f"Optimizer: SDG with learning rate={learning_rate} and regularization constant={reg_constant}")

#     return loss_fn, optimizer

def setupLocRegOptimizer(model, loss_version=1, learning_rate=1e-3, reg_constant=0, logfile=None):
    if loss_version == 2:
        loss_fn = LogReg_MSELoss_v2()

        # print("Optimizer: Single Reference MSE Loss Function")
        log(logfile, message="Optimizer: Single Reference MSE Loss Function ie LogReg_MSELoss_v2")
    
    else:
        loss_fn = LogReg_MSELoss()

        # print("Optimizer: Triple Reference MSE Loss Function")
        log(logfile, message="Optimizer: Triple Reference MSE Loss Function ie LogReg_MSELoss")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=reg_constant)
    # print(f"Optimizer: SDG with learning rate={learning_rate} and regularization constant={reg_constant}")
    log(logfile, message=f"Optimizer: SDG with learning rate={learning_rate} and regularization constant={reg_constant}")

    return loss_fn, optimizer

def setupLocRegOptimizer_v2(model, learning_rate=1e-3, reg_constant=0, logfile=None):
    """Goal is to test out, for eg., 
     - ADAM optim
     - MAE cost
     """
    pass

def collate_labelled_graphs(batch):
    collation = {}
    graphs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    for key in graphs[0].keys():
        collation[key] = [graph[key] for graph in graphs]

    return collation, torch.vstack(labels)

def train(dataloader, model, loss_fn, optimizer, 
        MSE_centroid_coeff=0, MSE_label_coeff=0, MSE_poolpt_coeff=0, 
        loss_version=1, logfile=None, testing_sample_size=None, 
        output_scaling=None):
    """ For single batch processing dataloaders"""
    size = len(dataloader.dataset)
    if testing_sample_size:
        print("testing_sample_size", testing_sample_size)

    model.train()
    train_set_loss = 0
    train_set_nominal_loss = 0
    for batch, (graph, label) in enumerate(dataloader):
        label = label.to(device)
        label = torch.squeeze(label, 0)

        keys = ("vertex_coord_list", "keypoint_indices_list", "edges_list")
        for key in keys:
            graph[key] = [torch.squeeze(tensor, 0).to(device) if tensor.shape[0] == 1 else tensor.to(device) for tensor in graph[key] ]

        pred = model(graph)
        pool_point = graph["vertex_coord_list"][-1]
        
        
        ## [NEEDS TO BE BETTER STRUCTURED/PLACED, so temporary]
        ## Standardizing sizes of various location
        pred = pred[:, :2]  ## should instead make the net output 2 or train fro 3 outputs including the rotation
        pool_point = pool_point[:, :2].type(torch.float)
        centroid = torch.mean(graph["vertex_coord_list"][0], dim=0, keepdim=True)[:, :2].type(torch.float)
        label = label[:, :2].type(torch.float)

        # ## DEBUG TOKEN
        # print("pred")
        # print(pred)
        # print(pred.dtype)
        # print("label")
        # print(label)
        # print(label.dtype)
        # print("centroid")
        # print(centroid)
        # print(centroid.dtype)
        # print("pool_point")
        # print(pool_point)
        # print(pool_point.dtype)
        # print()

        # ## DEBUG TOKEN: Checking for kwargs
        # print("MSE_centroid_coeff",MSE_centroid_coeff) 
        # print("MSE_label_coeff", MSE_label_coeff)
        # print("MSE_poolpt_coeff", MSE_poolpt_coeff)
        # print("loss_version", loss_version)
        # print("testing_sample_size", testing_sample_size)
        # print("output_scaling", output_scaling)

        
        if output_scaling == "sigmoid":
            # pred_sig = sigmoid(pred)
            # label_sig = sigmoid(label)
            # centroid_sig = sigmoid(centroid)
            # pool_sig = sigmoid(pool_point)

            pred = sigmoid(pred)
            label = sigmoid(label)
            centroid = sigmoid(centroid)
            pool_point = sigmoid(pool_point)
            # loss = MSE_label_coeff*loss_fn(pred_sig, label_sig) + MSE_centroid_coeff*loss_fn(pred_sig,centroid_sig ) + MSE_poolpt_coeff*loss_fn(pred_sig,pool_sig)
        
        if loss_version == 2:
            obj_loss = loss_fn(pred, label)*MSE_label_coeff

        else:
            obj_loss = loss_fn(pred, label, centroid, pool_point,
                                    coeff_label=MSE_label_coeff,
                                    coeff_centroid=MSE_centroid_coeff,
                                    coeff_pool_pt=MSE_poolpt_coeff)
            # loss_fMSE_label_coeff*loss_fn(pred, label) + MSE_centroid_coeff*loss_fn(pred,centroid ) + MSE_poolpt_coeff*loss_fn(pred,pool_point)        

        train_set_loss += obj_loss.item()
        nomimal_loss = torch.sum((pred - label)**2)**(0.5)
        train_set_nominal_loss += nomimal_loss

        # # ## DEBUG TOKEN
        # print("obj_loss.item()")
        # print("obj_loss",obj_loss)
        # print("nominal loss:", nomimal_loss)

        # Backpropagation
        optimizer.zero_grad()
        obj_loss.backward()
        optimizer.step()

        if batch % 100 == 0 or batch == size-1:
            loss, current = obj_loss.item(), (batch + 1) 
            avg_loss = train_set_loss/current
            # print(f"[Training] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            log(logfile, message=f"[Training] Sample Loss: {obj_loss:>7f},  Set Average:{avg_loss}  [{current:>5d}/{size:>5d}]")
        
        if testing_sample_size and batch == (testing_sample_size -1):
            break

    set_avg_loss = train_set_loss/size
    set_avg_nominal_loss = train_set_nominal_loss/size
    log(logfile, message=f"[Training] Loss: \n Training Set Avg loss: {set_avg_loss:>8f} \n")
    log(logfile, message=f"[Training] Loss: \n Training Set Avg nominal loss: {set_avg_nominal_loss:>8f} \n")

    return set_avg_loss, set_avg_nominal_loss

def train_v2(dataloader, model, loss_fn, optimizer, logfile=None):
    """ For the case where dataloader has an appropriate collate_fn to process batches"""
    size = len(dataloader.dataset)
    model.train()
    train_set_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        y = y.to(device)
        if y.shape[0] == 1:
            y = torch.squeeze(y, 0)

        keys = ("vertex_coord_list", "keypoint_indices_list", "edges_list")
        for key in keys:
            X[key] = [torch.squeeze(tensor, 0).to(device) if tensor.shape[0] == 1 else tensor.to(device) for tensor in X[key] ]

        pred = model(X)
        loss = loss_fn(pred, y)
        train_set_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0 or batch == size-1:
            loss, current = loss.item(), (batch + 1) 
            # print(f"[Training] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            log(logfile, message=f"[Training] Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    log(logfile, message=f"[Training] Loss: \n Training Set loss: {train_set_loss:>8f} \n")

    return train_set_loss


def batch_train(dataloader, batch_size, model, loss_fn, optimizer, logfile=None):
    """
    Divide set into batches 
    Aggregate loss across a batch, average out, then grad descent
    """
    size = len(dataloader.dataset)
    model.train()
    train_set_loss = 0
    batch_losses = []
    for idx, (X, y) in enumerate(dataloader):
        y = y.to(device)
        y = torch.squeeze(y, 0)

        keys = ("vertex_coord_list", "keypoint_indices_list", "edges_list")
        for key in keys:
            X[key] = [torch.squeeze(tensor, 0).to(device) for tensor in X[key]]

        pred = model(X)
        loss = loss_fn(pred, y)
        train_set_loss += loss.item()
        batch_losses.append(loss)
        
        if idx % batch_size == batch_size-1 or idx == batch_size-1:

            ## effect backprop update here
            avg_batch_loss = sum(batch_losses)/len(batch_losses)
            avg_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ## then reset
            ### DEBUG TOKEN
            # print(len(batch_losses))
            batch_losses = []

        if idx % 100 == 0:
            loss, current = loss.item(), (idx + 1) 
            log(logfile, message=f"[Training] Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    log(logfile, message=f"[Training] Loss: \n Training Set loss: {train_set_loss:>8f} \n")

    return train_set_loss

def test(dataloader, model, loss_fn, logfile=None, epoch="", 
        return_plots=False, draw_list_by_idx=None, 
        testing_sample_size=None, 
        MSE_centroid_coeff=0, MSE_label_coeff=0, MSE_poolpt_coeff=0, 
        loss_version=1, output_scaling=None):
    

    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()
    test_set_loss = 0
    test_set_nominal_loss = 0

    plots = []
    if size >= 5:
        draw_list = list(np.linspace(0,size-1,5).astype(int))
    else:
        draw_list = list(np.linspace(0,size-1,size).astype(int))
    if draw_list_by_idx:
        draw_list = draw_list_by_idx
    
    # ## DEBUG TOKEN : same predictions
    # graphs = {}

    with torch.no_grad():
        for i, (graph, label) in enumerate(dataloader):
            label = label.to(device)
            label = torch.squeeze(label, 0)

            keys = ("vertex_coord_list", "keypoint_indices_list", "edges_list")
            for key in keys:
                graph[key] = [torch.squeeze(tensor,0).to(device) for tensor in graph[key]]
            
            # ## DEBUG TOKEN : same predictions
            # ## Check that the graph inputs are different
            # print("graph")
            # print(graph)
            # graphs[i] = graph

            pred = model(graph)
            pool_point = graph["vertex_coord_list"][-1]

            ## [NEEDS TO BE BETTER STRUCTURED/PLACED, so temporary]
            ## Standardizing sizes of various location
            pred = pred[:, :2]  ## should instead make the net output 2 or train fro 3 outputs including the rotation
            pool_point = pool_point[:, :2].type(torch.float)
            centroid = torch.mean(graph["vertex_coord_list"][0], dim=0, keepdim=True)[:, :2].type(torch.float)
            label = label.type(torch.float)[:, :2]

            if output_scaling == "sigmoid":
                pred = sigmoid(pred)
                label = sigmoid(label)
                centroid = sigmoid(centroid)
                pool_point = sigmoid(pool_point)

            # sample_loss = loss_fn(pred, label).item()
            # test_set_loss += sample_loss
            
            if loss_version == 2:
                obj_loss = loss_fn(pred, label)*MSE_label_coeff

            else:
                obj_loss = loss_fn(pred, label, centroid, pool_point,
                                    coeff_label=MSE_label_coeff,
                                    coeff_centroid=MSE_centroid_coeff,
                                    coeff_pool_pt=MSE_poolpt_coeff)

            sample_loss = obj_loss.item()
            sample_nominal_loss = torch.sum((pred - label)**2)**(0.5)
            
            test_set_loss += sample_loss
            test_set_nominal_loss +=sample_nominal_loss

            # # # ## DEBUG TOKEN
            # print("pred")
            # print(pred)
            # # print(pred.dtype)
            # print("label")
            # print(label)
            # # print(label.dtype)
            # # print("centroid")
            # # print()
            # print("sample_loss")
            # print(sample_loss)
            # print("sample_nominal_loss")
            # print(sample_nominal_loss)
            # print()

            if return_plots and i in draw_list:
                summary = f"Sample: {i}, Epoch: {epoch} -->Sample Obj Loss: {sample_loss} Nominal Loss:{sample_nominal_loss}"
                plot = plot_inference_loc(pred, graph, cuda= device=="cpu", additional_pts=True, title=summary, caption=summary ) ## might have to remove from GPUs to draw
                plots.append(plot)

            if testing_sample_size and i == (testing_sample_size -1) :
                break
    
    # ## DEBUG TOKEN : same predictions
    # ## Check that the graph inputs are different
    
    # print("equality check")
    # print("graphs[0]== graphs[1]")
    # print(graphs[0]== graphs[1])

    if testing_sample_size:
        test_set_avg_loss = test_set_loss/testing_sample_size
        test_set_avg_nominal_loss = test_set_nominal_loss/testing_sample_size
    else:
        test_set_avg_loss = test_set_loss/size
        test_set_avg_nominal_loss = test_set_nominal_loss/size

    # print(f"[Testing] Error: \n Avg loss: {test_loss:>8f} \n")
    log(logfile, message=f"[Testing] Error: \n Test Set Avg loss: {test_set_avg_loss:>8f}\n")
    log(logfile, message=f"[Testing] Error: \n Test Set Avg Nominal loss: {test_set_avg_nominal_loss:>8f}\n")
    

    if return_plots:
        return test_set_avg_loss, test_set_avg_nominal_loss, plots

    return test_set_avg_loss, test_set_avg_nominal_loss

#### **Packaged as function for use in test py notebooks
def run_epochs(model, loss_fn, optimizer, train_dataloader, test_dataloader=None, epochs=1,\
     MSE_centroid_coeff=1, MSE_label_coeff=1, model_params_path=None, logfile=None):
    training_start = datetime.now()
    training_start_str = datetime.strftime(training_start, '%Y_%m_%d_%H_%M_%S')
    log(logfile, message=f"Started Training: {training_start_str}")

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        log(logfile, message=f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, MSE_centroid_coeff, MSE_label_coeff, logfile=logfile)
        
        if test_dataloader:
            test(test_dataloader, model, loss_fn, logfile=logfile)
    
    training_end = datetime.now()
    training_end_str = datetime.strftime(training_end, '%Y_%m_%d_%H_%M_%S')
    log(logfile, message=f"Finished Training: {training_end_str}")
    
    training_duration_str = datetime.strftime(training_end - training_start, '%Y_%m_%d_%H_%M_%S')
    log(logfile, "Training Duration: {training_duration_str}")
    log(logfile, message="Done!")

    if model_params_path:
        torch.save(model.state_dict(), model_params_path)
        # print(f"Saved PyTorch Model State to {model_params_path}")
        log(logfile, message=f"Saved PyTorch Model State to {model_params_path}")

#### **Packaged as function for use in test py notebooks
def setup_and_run(epochs, model=None, reg_constant=0, learning_rate=1e-3, MSE_centroid_coeff=1, MSE_label_coeff=1, from_model_params_path=None, to_model_params_path=None, logfile=None):
    
    ## The Data
    train_dataloader, test_dataloader = setupDataLoaders(train_root_dir, test_root_dir, logfile=logfile)

    ## The Model
    if not model:
        model = setupLogRegModel(model_params_path=from_model_params_path, logfile=logfile)

    ## The Objective Optimizer
    loss_fn, optimizer = setupLocRegOptimizer(model, reg_constant=reg_constant, learning_rate=learning_rate, logfile=logfile)

    if to_model_params_path:
        model_params_path = to_model_params_path
    else:
        model_params_path = get_filename(model_params_test_dir, model_params_tag)
    
    ## Train and Test through epochs
    run_epochs(model, loss_fn, optimizer, train_dataloader, test_dataloader, epochs=epochs, \
        MSE_centroid_coeff=MSE_centroid_coeff, MSE_label_coeff=MSE_label_coeff, \
        model_params_path=model_params_path, logfile=logfile)

    return model

