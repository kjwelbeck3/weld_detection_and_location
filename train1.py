
import os
import torch
from datetime import datetime
import random
from dataset import MyDataset
from Instseg_model import MultiLayerFastLocalGraphModelV2
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

def log(logfile=None, message=""):
    """
    Utility function to timestamp print statements
    Optionally, also print to file object
    """
    stamp = datetime.now()
    str = f"[{stamp}] {message}"
    print(str)
    if logfile:
        logfile.write("\n")
        logfile.write(str)
        logfile.flush()
    
def plot_inference(predictions, points, cuda, title="", caption=""):
    """
    Produce 2D matplotlib fig of graph points with color-coded predictions 
    """
    preds_cpu, points_cpu = None, None
    if cuda:
        preds_cpu = predictions.cpu().detach().numpy()
        points_cpu = points.cpu().detach().numpy()
    else:
        preds_cpu = predictions.detach().numpy()
        points_cpu = points.detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.scatter(points_cpu[:,0], points_cpu[:,1],s=0.25,c=preds_cpu, cmap = "Reds")
    ax.set_aspect(1)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    legend = ax.legend(*im.legend_elements(), bbox_to_anchor=(1.1, 1), loc="upper right" )
    ax.add_artist(legend)

    ax.text(0.5, -0.25, caption, style='italic', \
		horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

    return fig

def plot_inference_loc(prediction, graph, cuda, additional_pts=False, title="", caption=""):
    """
    Produce 2D matplotlib fig of graph points with prediction overlay 
    """
    preds_cpu, points_cpu, pool_pt_cpu = None, None, None
    points = graph["vertex_coord_list"][0]
    pool_point = graph["vertex_coord_list"][-1]
    
    

    if cuda and torch.cuda.is_available():
        preds_cpu = prediction.cpu().detach().numpy()
        points_cpu = points.cpu().detach().numpy()
        pool_pt_cpu = points.cpu().detach().numpy()
    else:
        if not isinstance(prediction, np.ndarray):
            preds_cpu = prediction.detach().numpy()
        else:
            preds_cpu = prediction

        if not isinstance(points, np.ndarray):
            points_cpu = points.detach().numpy()
            pool_pt_cpu = points.detach().numpy()

        else:
            points_cpu = points
            pool_pt_cpu = pool_point
 
    centroid = np.mean(points_cpu, axis=0)

    # ## DEBUG TOKEN
    # print("points", points_cpu)
    # print("pool_point", pool_pt_cpu)
    # print("centroid", centroid)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.scatter(points_cpu[:,0], points_cpu[:,1], color="blue", label="point cloud") ## ,s=0.25, cmap = "Reds")
    ax.scatter(preds_cpu[:,0], preds_cpu[:,1], color="black", label="prediction") ## ,s=0.25,c=preds_cpu, cmap = "Reds")

    if additional_pts:
        ax.scatter(pool_point[:, 0], pool_point[:, 1], color="yellow", label="pool point", marker="D")
        ax.scatter(centroid[0], centroid[1], color="green", label="centroid", marker="X" )

    ax.set_aspect(1)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    # ax

    # legend = ax.legend(*im.legend_elements(), bbox_to_anchor=(1.1, 1), loc="upper right" )
    # ax.add_artist(legend)
    ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right")

    ax.text(0.5, -0.25, caption, style='italic', \
		horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

    return fig

def run():
    LR = 1e-4
    epochs = 50
    # epochs = 2
    nametag ="train1"
    testset_idxs_to_draw_through_training = [0,1,2,3,4] # TODO Remember to increase the set
    draw_list = testset_idxs_to_draw_through_training
    model = MultiLayerFastLocalGraphModelV2(num_classes=3,
                max_instance_no=7, mode='train')
    dataset_root = "./_data/"
    cuda = True
    cuda = cuda and torch.cuda.is_available()

    _logfile_dir = "./_log/logfiles/"
    if not os.path.isdir(_logfile_dir):
        print(f"Logfile directory,  {_logfile_dir}, does not exist. ")
        print(f"Creating Logfile directory,  {_logfile_dir}, ...")
        os.mkdir(_logfile_dir)
        if os.path.isdir(_logfile_dir):
            print("Logfile directory created")
        else:
            print("Could not create Logfile directory")

    starttime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    _logfile_path = f"{_logfile_dir}{nametag}_{starttime}.txt"
    _logfile = open(_logfile_path, 'w')


    # cuda = False
    # print("Cuda available", torch.cuda.is_available())
    # print("Cuda ", cuda)

    if cuda:
        torch.cuda.empty_cache()

    log(_logfile, f"Start time: {starttime}")
    log(_logfile, f"Cuda available: {torch.cuda.is_available()}")
    log(_logfile, f"Cuda: {cuda}")


    # print()
    # print("Model:")
    # print(model)

    log(_logfile)
    log(_logfile, "Model:")
    log(_logfile, model)

    if cuda:         
        model = model.cuda()

    ## Change: to load the train-test split from a text files 
    train_set, test_set= [], []

    # print()
    # print(f"Verifying dataset root directory, {dataset_root}:", os.path.isdir(dataset_root))
    # print(f"Verifying train split file, {dataset_root}train.txt: ", os.path.isfile(dataset_root+"train.txt"))
    # print(f"Verifying test split file, {dataset_root}test.txt: ", os.path.isfile(dataset_root+"test.txt"))
    # print()

    log(_logfile)
    log(_logfile, f"Verifying dataset root directory, {dataset_root}: {os.path.isdir(dataset_root)}")
    log(_logfile, f"Verifying train split file, {dataset_root}train.txt: {os.path.isfile(dataset_root+'train.txt')}")
    log(_logfile, f"Verifying test split file, {dataset_root}test.txt: {os.path.isfile(dataset_root+'test.txt')}")
    log(_logfile)

    if os.path.isfile(dataset_root+"train.txt") and os.path.isfile(dataset_root+"test.txt"):
        with open(dataset_root+"train.txt", "r") as traintext:
            samples = traintext.readlines()
            for sample in samples:
                # train_set.append(eval(sample))
                _sample = [dataset_root + a[2:] for a in eval(sample)]
                train_set.append(_sample)

        with open(dataset_root+"test.txt", "r") as testtext:
            samples = testtext.readlines()
            for sample in samples:
                # test_set.append(eval(sample))
                _sample = [dataset_root + a[2:] for a in eval(sample)]
                test_set.append(_sample)

    ## For Debug
    # print(train_set)
    # print(test_set)
    # log(_logfile, train_set)
    # log(_logfile, test_set)

    train_data = MyDataset(dataset=train_set)
    train_loader = data_utils.DataLoader(dataset=train_data, batch_size=1, shuffle=True)#, shuffle=True, num_workers=0)

    test_data=MyDataset(dataset=test_set)
    test_loader = data_utils.DataLoader(dataset=test_data, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_start = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S') 
    writer = SummaryWriter(f'runs/GNN_expt_{nametag}_'+training_start)
    model_directory = f'./_model/{nametag}/{training_start}/'
    if not os.path.isdir(model_directory):

        # print("Could not find model directory, ", model_directory)
        # print("Attempting to create...")
        log(_logfile, f"Could not find model directory, {model_directory}")
        log(_logfile, "Attempting to create...")

        os.makedirs(model_directory)

        if os.path.isdir(model_directory):
            # print(f"created: {model_directory}")
            log(_logfile, f"created: {model_directory}")
        else:
            # print(f"could not create: {model_directory}")
            log(_logfile, f"could not create: {model_directory}")

    log_directory = f"./_log/{nametag}/{training_start}/"
    if not os.path.isdir(log_directory):
        # print(f"Could not find log directory, {log_directory}")
        # print(f"Attempting to create...")
        log(_logfile, f"Could not find log directory, {log_directory}")
        log(_logfile, f"Attempting to create...")

        os.makedirs(log_directory)
        if os.path.isdir(log_directory):
            # print(f"created: {log_directory}")
            log(_logfile, f"created: {log_directory}")
        else:
            # print(f"could not create: {log_directory}")
            log(_logfile, f"could not create: {log_directory}")

    train_log_file = f"{log_directory}{training_start}_trainloss.txt"
    test_log_file = f"{log_directory}{training_start}_testloss.txt"

    with open(train_log_file, "w") as train_log:
        train_log.write("epoch, step, tot_loss, cls_acc, inst_acc\n")

    with open(test_log_file, "w") as test_log:
        test_log.write("epoch, step, tot_loss, cls_acc, inst_acc\n")

    min_test_loss_model_param = None
    min_test_loss = 100000
    min_test_loss_epoch = None
    min_test_loss_accuracy = None

    ## For Logging the network to Tensorboard
    batch_zero = train_data[0]
    new_batch_zero = []
    for item in batch_zero:
        if not isinstance(item, torch.Tensor):
            item = [torch.squeeze(x,0).cuda() if cuda else torch.squeeze(x, 0) for x in item]
        else: 
            item = torch.squeeze(item,0).cuda() if cuda else torch.squeeze(item, 0)
        new_batch_zero += [item]
        
    vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, inst_labels = new_batch_zero
    batch_zero = new_batch_zero

    writer.add_graph(model, batch_zero, True)

    for epoch in range(epochs):
    # for epoch in range(1):
        ## Looping through the training samples
        last_step = 0
        running_loss = 0.0
        running_reg_loss = 0.0
        running_seg_loss = 0.0
        running_cls_loss = 0.0
        running_epoch_loss = 0.0
        running_cls_accuracy = 0
        running_inst_accuracy = 0
        for step, (vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, inst_labels) in enumerate(train_loader, 1):
            
            batch = (vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, inst_labels)
            
        ## Activate cuda processing ie Send to GPUs, if specified
            new_batch = []
            for item in batch:
                if not isinstance(item, torch.Tensor):
                    item = [torch.squeeze(x,0).cuda() if cuda else torch.squeeze(x, 0) for x in item]
                else: 
                    item = torch.squeeze(item,0).cuda() if cuda else torch.squeeze(item, 0)
                new_batch += [item]
                
            vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, inst_labels = new_batch  
            batch = new_batch

        ## Forward pass     
            logits, inst_seg = model(*batch)#, is_training=True)     
        
        ## From logits to classification for each vertex 
            cls_predictions = torch.argmax(logits, dim=1)
            seg_predictions = torch.argmax(inst_seg, dim=1) 
            
        ## Calculating focal loss on class and instance predictions; 1:1:1 weighting by default        
            loss_dict = model.loss(logits, cls_labels, inst_seg, inst_labels)
            accuracy_dict = model.accuracy(cls_predictions, cls_labels, seg_predictions, inst_labels)
            t_cls_loss, t_seg_loss, t_reg_loss = loss_dict['cls_loss'], loss_dict['seg_loss'], loss_dict['reg_loss']
            running_reg_loss+=t_reg_loss
            running_seg_loss+=t_seg_loss
            running_cls_loss+=t_cls_loss

            t_total_loss = t_cls_loss + 10*t_seg_loss + t_reg_loss  
            # t_total_loss = t_cls_loss + t_seg_loss + t_reg_loss
            running_loss += t_total_loss
            running_epoch_loss += t_total_loss
            last_step = step

            cls_acc = accuracy_dict['cls_accuracy']
            inst_acc = accuracy_dict['inst_accuracy']
            running_cls_accuracy += cls_acc
            running_inst_accuracy += inst_acc
            # print(f"step: {step}")
            # print(f"step losses: {t_cls_loss} +  {t_seg_loss} + {t_reg_loss} = {t_cls_loss + t_seg_loss + t_reg_loss}")
            # print(f"accuracies: {cls_acc}, {inst_acc}")

        ## Logging for inspection/debug
            with open(train_log_file, "a") as train_log:
                train_log.write(f"{epoch}, {step}, {t_total_loss}, {cls_acc}, {inst_acc}\n")

        ## Updating model parameters
            optimizer.zero_grad()
            t_total_loss.backward()
            optimizer.step()

        ## Averaging, logging and resetting losses after every 20 samples
            if step % 20 == 0:
            # if step % 2 == 0:
                running_loss /= 20.0
                running_reg_loss /= 20.0
                running_seg_loss /= 20.0
                running_cls_loss /= 20.0
                # print(f"#{epoch}.{step} -> [seg_loss: {running_seg_loss}; cls_loss: {running_cls_loss}; reg_loss: {running_reg_loss}")
                log(_logfile, f"#{epoch}.{step} -> [seg_loss(last20): {running_seg_loss}; cls_loss(last20): {running_cls_loss}; reg_loss(last20): {running_reg_loss}; combo_loss(last20): {running_loss}")
                writer.add_scalar('train/combo_loss_after20', running_loss, epoch*len(train_loader) + step )
                writer.add_scalar('train/reg_loss_after20', running_reg_loss, epoch*len(train_loader) + step )
                writer.add_scalar('train/seg_loss_after20', running_seg_loss, epoch*len(train_loader) + step )
                writer.add_scalar('train/cls_loss_after20', running_cls_loss, epoch*len(train_loader) + step )
                running_loss, running_reg_loss, running_reg_loss, running_cls_loss = 0.0, 0.0, 0.0, 0.0

            log(_logfile, f"Training Step {step}... Done")

        ## Logging the training loss averaged across the epoch
        writer.add_scalar('train/loss_per_epoch', running_epoch_loss/last_step, epoch)
        writer.add_scalar('train/cls_accuracy_per_epoch', running_cls_accuracy/last_step, epoch)
        writer.add_scalar('train/inst_accuracy_per_epoch', running_inst_accuracy/last_step, epoch)
        
        
        
        ## To calculate the corresponding test loss per epoch, for all epochs
        ## Note: the model has been updated at the point, so the labeled test loss has to be approptiately forward anotated
        last_test_step = 0
        test_loss = 0
        test_cls_accuracy = 0
        test_inst_accuracy = 0
        for step, (vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, inst_labels) in enumerate(test_loader, 1):
            
            batch = (vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, inst_labels)
            
        ## Activate cuda processing ie Send to GPUs, if specified
            new_batch = []
            for item in batch:
                if not isinstance(item, torch.Tensor):
                    item = [torch.squeeze(x,0).cuda() if cuda else torch.squeeze(x, 0) for x in item]
                else: 
                    item = torch.squeeze(item,0).cuda() if cuda else torch.squeeze(item, 0)
                new_batch += [item]
                
            vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, inst_labels = new_batch  
            batch = new_batch

        ## Forward pass    
            logits, inst_seg = None, None 
            with torch.no_grad():
                logits, inst_seg = model(*batch)#, is_training=False)
        
        ## From logits to classification for each vertex 
            cls_predictions = torch.argmax(logits, dim=1)
            seg_predictions = torch.argmax(inst_seg, dim=1) 
        
        ## Calculating focal loss on class and instance predictions; 1:1:1 weighting by default
            loss_dict = model.loss(logits, cls_labels, inst_seg, inst_labels)
            accuracy_dict = model.accuracy(cls_predictions, cls_labels, seg_predictions, inst_labels)
            t_cls_loss, t_seg_loss, t_reg_loss = loss_dict['cls_loss'], loss_dict['seg_loss'], loss_dict['reg_loss']
            t_total_loss = t_cls_loss + t_seg_loss + t_reg_loss
            test_loss += t_total_loss
            last_test_step = step 
        
        ## Computing accuracy
            cls_acc = accuracy_dict['cls_accuracy']
            inst_acc = accuracy_dict['inst_accuracy']
            test_cls_accuracy += cls_acc
            test_inst_accuracy += inst_acc
        
        ## Logging for inspection/debug
            with open(test_log_file, "a") as test_log:
                test_log.write(f"{epoch}, {step}, {t_total_loss}, {cls_acc}, {inst_acc}\n")

        ## Saving plots for subset of test set periodically:
            if epoch % 10 == 9 and step in draw_list:
            # if epoch % 10 == 0 and step in draw_list:
                cls_fig = plot_inference(cls_predictions, vertex_coord_list[2], cuda, f"cls{step}_epoch{epoch} -> cls/total loss: {t_cls_loss}/{t_total_loss}")
                inst_fig = plot_inference(seg_predictions, vertex_coord_list[2], cuda, f"inst{step}_epoch{epoch} -> cls/total loss: {t_seg_loss}/{t_total_loss}")
                writer.add_figure('test/samples/cls{step}', cls_fig, epoch)
                writer.add_figure('test/samples/inst{step}', inst_fig, epoch)

            log(_logfile, f"Test Step {step}... Done")

        ## Tracking the test loss per epoch
        writer.add_scalar('test/loss_per_epoch/', test_loss/last_test_step, epoch+1)
        writer.add_scalar('test/cls_acc_per_epoch/', test_cls_accuracy/last_test_step, epoch+1)
        writer.add_scalar('test/inst_acc_per_epoch/', test_inst_accuracy/last_test_step, epoch+1)


        if test_loss < min_test_loss:
            min_test_loss = test_loss
            # min_test_loss_model_param = model.parameters()
            min_test_loss_model_param = model.state_dict()
            min_test_loss_epoch = epoch
            # print("new min_test_lost")
            log(_logfile, "new min_test_lost")

        ## Saving the model parameters periodically 
        if epoch % 10 == 9:
        # if epoch % 10 == 0:
            timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
            chkpt_path =  model_directory+"params_epoch"+str(epoch)+"_"+timestamp+".pt"
            torch.save(model.state_dict(), chkpt_path)
            # print(f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
            # print(f"[{timestamp}]:  - epoch = {epoch}")
            # print(f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
            # print()

            log(_logfile, f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
            log(_logfile, f"[{timestamp}]:  - epoch = {epoch}")
            log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
            log(_logfile)

        log(_logfile, f"Epoch {epoch}... Done")

    ## Saving the model parameters of the minimized test loss and the last epoch 
    timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S') 
    min_path = model_directory+"params_epoch"+str(min_test_loss_epoch)+"_for_min_test_loss.pt"
    chkpt_path =  model_directory+"params_epoch"+str(epoch)+"_"+timestamp+".pt"

    torch.save(model.state_dict(), chkpt_path)
    # print(f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
    # print(f"[{timestamp}]:  - epoch = {epoch}")
    # print(f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
    # print()

    log(_logfile, f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
    log(_logfile, f"[{timestamp}]:  - epoch = {epoch}")
    log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
    log(_logfile)

    torch.save(min_test_loss_model_param, min_path)
    # print(f"[{timestamp}]: Saved min_test_loss_model_params - {min_path}")
    # print(f"[{timestamp}]:  - min_test_loss = {min_test_loss}")
    # print(f"[{timestamp}]:  - min_test_loss_epoch = {min_test_loss_epoch}")
    # print()

    log(_logfile, f"[{timestamp}]: Saved min_test_loss_model_params - {min_path}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss = {min_test_loss}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss_epoch = {min_test_loss_epoch}")
    log(_logfile)

    stoptime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log(_logfile, f"Stop time: {stoptime}")
    log(_logfile)
    log(_logfile, "Closing logfile")
    _logfile.close()
