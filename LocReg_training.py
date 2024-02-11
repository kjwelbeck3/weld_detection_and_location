import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from LocReg_model import LocationRegressionModelV1
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
def setupDataLoaders(train_root_dir, test_root_dir=None, batch_size=1, logfile=None):
    
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
    train_dataloader = DataLoader(training_dataset, batch_size=1)

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
def setupLocRegOptimizer(model, learning_rate=1e-3, reg_constant=0, logfile=None):
    loss_fn = nn.MSELoss()

    # print("Optimizer: MSE Loss Function")
    log(logfile, message="Optimizer: MSE Loss Function")

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

def train(dataloader, model, loss_fn, optimizer, logfile=None):
    """ For single batch processing dataloaders"""
    size = len(dataloader.dataset)
    model.train()
    train_set_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        y = y.to(device)
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

def test(dataloader, model, loss_fn, logfile=None, epoch="", return_plots=False, draw_list_by_idx=None ):
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_set_loss = 0

    plots = []
    if size >= 5:
        draw_list = list(np.linspace(0,size-1,5).astype(int))
    else:
        draw_list = list(np.linspace(0,size-1,size).astype(int))
    if draw_list_by_idx:
        draw_list = draw_list_by_idx

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            y = y.to(device)
            y = torch.squeeze(y, 0)

            keys = ("vertex_coord_list", "keypoint_indices_list", "edges_list")
            for key in keys:
                X[key] = [torch.squeeze(tensor,0).to(device) for tensor in X[key]]

            pred = model(X)
            sample_loss = loss_fn(pred, y).item()
            test_set_loss += sample_loss

            if return_plots and i in draw_list:
                summary = f"Sample: {i}, Epoch: {epoch} --> Sample Loss: {sample_loss}"
                plot = plot_inference_loc(pred, X, cuda= device=="cpu", title=summary, caption=summary ) ## might have to remove from GPUs to draw
                plots.append(plot)

    test_set_loss /= num_batches
    # print(f"[Testing] Error: \n Avg loss: {test_loss:>8f} \n")
    log(logfile, message=f"[Testing] Error: \n Test Set loss: {test_set_loss:>8f} \n")

    if return_plots:
        return test_set_loss, plots

    return test_set_loss

#### **Packaged as function for use in test py notebooks
def run_epochs(model, loss_fn, optimizer, train_dataloader, test_dataloader=None, epochs=1, model_params_path=None, logfile=None):
    training_start = datetime.now()
    training_start_str = datetime.strftime(training_start, '%Y_%m_%d_%H_%M_%S')
    log(logfile, message=f"Started Training: {training_start_str}")

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        log(logfile, message=f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, logfile=logfile)
        
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
def setup_and_run(epochs, model=None, from_model_params_path=None, to_model_params_path=None, logfile=None):
    
    ## The Data
    train_dataloader, test_dataloader = setupDataLoaders(train_root_dir, test_root_dir, logfile=logfile)

    ## The Model
    if not model:
        model = setupLogRegModel(model_params_path=from_model_params_path, logfile=logfile)

    ## The Objective Optimizer
    loss_fn, optimizer = setupLocRegOptimizer(model, logfile=logfile)

    if to_model_params_path:
        model_params_path = to_model_params_path
    else:
        model_params_path = get_filename(model_params_test_dir, model_params_tag)
    
    ## Train and Test through epochs
    run_epochs(model, loss_fn, optimizer, train_dataloader, test_dataloader, epochs=epochs, model_params_path=model_params_path, logfile=logfile)

    return model

