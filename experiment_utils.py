import torch
import matplotlib
import os
from train1 import log, plot_inference, plot_inference_loc
from datetime import datetime
from dataset import MyDataset
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from LocReg_training import batch_train, setupDataLoaders, setupLogRegModel, setupLocRegOptimizer, run_epochs, setup_and_run, train, test


def train_model(nametag, model, dataset_root="./_data/", log_root="./_log/", batch_size=1, learning_rate=1e-4, epochs=150, draw_list=[0,1,2,3,4], cuda=True, initializing_state_dict=None, debug=False, train_data_subset=None, test_data_subset=None):
   
    ## Printing Function Arguments for inspection
    print("####")
    print(f"nametag: {nametag}")
    print(f"dataset_root: {dataset_root}")
    print(f"log_root: {log_root}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"epiochs: {epochs}")
    print(f"draw_list: {draw_list}")
    print(f"cuda: {cuda}")
    print("####")
   

    ## Setting up file for logging timestamped events
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


    ## Setting up cuda and initializing model params, if applicable
    if cuda:
        torch.cuda.empty_cache()

    log(_logfile, f"Start time: {starttime}")
    log(_logfile, f"Cuda available: {torch.cuda.is_available()}")
    log(_logfile, f"Cuda: {cuda}")

    if initializing_state_dict:
        log(_logfile, f"Loading previously saved model parameters from {initializing_state_dict} ...")
        model.load_state_dict(torch.load(initializing_state_dict))
        log(_logfile, "Loaded previously saved model parameters.")

    log(_logfile)
    log(_logfile, "Model:")
    log(_logfile, model)

    if cuda:         
        model = model.cuda()


    ## Organizing list of training samples and list of training samples from respective txt files
    ## [TO CHANGE]: to load the train-test split from a common source/file, instead of pre-separated splits
    train_set, test_set= [], []

    log(_logfile)
    log(_logfile, f"Verifying dataset root directory, {dataset_root}: {os.path.isdir(dataset_root)}")
    log(_logfile, f"Verifying train split file, {dataset_root}train.txt: {os.path.isfile(dataset_root+'train.txt')}")
    log(_logfile, f"Verifying test split file, {dataset_root}test.txt: {os.path.isfile(dataset_root+'test.txt')}")
    log(_logfile)

    if os.path.isfile(dataset_root+"train.txt") and os.path.isfile(dataset_root+"test.txt"):
        with open(dataset_root+"train.txt", "r") as traintext:
            samples = traintext.readlines()
            sample_count = len(samples)
            if train_data_subset:
                sample_count = train_data_subset
            for sample in samples[:sample_count]:
                # train_set.append(eval(sample))
                # _sample = [dataset_root + a[2:] for a in eval(sample)]
                _sample = eval(sample)
                train_set.append(_sample)

        with open(dataset_root+"test.txt", "r") as testtext:
            samples = testtext.readlines()
            sample_count = len(samples)
            if test_data_subset:
                sample_count = test_data_subset
            for sample in samples[:sample_count]:
                # test_set.append(eval(sample))
                # _sample = [dataset_root + a[2:] for a in eval(sample)]
                _sample = eval(sample)
                test_set.append(_sample)
    else:
        log(_logfile, f"train.txt and/or test.txt does not exist in the dataset root, {dataset_root}.")
        log(_logfile, f"Exiting training session.")
        return None


    ## For Debug
    if debug:
        log(_logfile, "[DEBUG] train_set")
        log(_logfile, train_set)
        log(_logfile, "[DEBUG] test_set:")
        log(_logfile, test_set)


    ## Organizing datasets, dataloaders, optimizer
    ## [TO CHANGE]:Revise MyDataset to allow for mini-batch loading
    train_data = MyDataset(dataset=train_set)
    train_loader = data_utils.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)#, shuffle=True, num_workers=0)

    test_data=MyDataset(dataset=test_set)
    test_loader = data_utils.DataLoader(dataset=test_data, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    ## Setting up files for tensorboarding, logging training events, and saving model parameters
    training_start = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S') 
    writer = SummaryWriter(f'runs/GNN_expt_{nametag}_'+training_start)
    model_directory = f'./_model/{nametag}/{training_start}/'
    if not os.path.isdir(model_directory):

        log(_logfile, f"Could not find model directory, {model_directory}")
        log(_logfile, "Attempting to create...")

        os.makedirs(model_directory)

        if os.path.isdir(model_directory):
            log(_logfile, f"created: {model_directory}")
        else:
            log(_logfile, f"could not create: {model_directory}")

    log_directory = f"{log_root}{nametag}/{training_start}/"
    if not os.path.isdir(log_directory):
        log(_logfile, f"Could not find log directory, {log_directory}")
        log(_logfile, f"Attempting to create...")

        os.makedirs(log_directory)
        if os.path.isdir(log_directory):
            log(_logfile, f"created: {log_directory}")
        else:
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
    # min_test_loss_accuracy = None
    min_test_loss_accuracy_cls = None
    min_test_loss_accuracy_inst = None


    ## For Logging the graph to Tensorboard [TO REVISIT]
    # batch_zero = train_data[0]
    # new_batch_zero = []
    # for item in batch_zero:
    #     if not isinstance(item, torch.Tensor):
    #         item = [torch.squeeze(x,0).cuda() if cuda else torch.squeeze(x, 0) for x in item]
    #     else: 
    #         item = torch.squeeze(item,0).cuda() if cuda else torch.squeeze(item, 0)
    #     new_batch_zero += [item]
        
    # vertex_coord_list, keypoint_indices_list, edges_list, \
    #         cls_labels, inst_labels = new_batch_zero
    # batch_zero = new_batch_zero

    # # writer.add_graph(model, batch_zero, True)


    ## For each epoch, loop through training on samples,  tally up losses and log events.
    ## For each sample, send to GPU if applicable, forward-pass through model, extract segmentation classes, compute loss, optimize objective via SGD
    ## For each epoch, also loops through test samples for logging/tensorboarding predictions and losses, and periodically saves model params  
    for epoch in range(epochs):
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
            # logits, inst_seg = model(batch)#, is_training=True)     
        
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
                # train_log.write(f"{epoch}, {step}, {t_total_loss}") #", {cls_acc}, {inst_acc}\n")
                train_log.write(f"{epoch}, {step}, {t_total_loss}, {cls_acc}, {inst_acc}\n")

            ## Updating model parameters
            optimizer.zero_grad()
            t_total_loss.backward()
            optimizer.step()

            ## Averaging, logging and resetting losses after every 20 samples
            if step % 20 == 0:
                running_loss /= 20.0
                running_reg_loss /= 20.0
                running_seg_loss /= 20.0
                running_cls_loss /= 20.0
            
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
                # logits, inst_seg = model(batch)#, is_training=False)
        
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
                # test_log.write(f"{epoch}, {step}, {t_total_loss}")#", {cls_acc}, {inst_acc}\n")
                test_log.write(f"{epoch}, {step}, {t_total_loss}, {cls_acc}, {inst_acc}\n")

        ## Saving plots for subset of test set periodically:
            if epoch % 10 == 9 and step in draw_list:
            # if epoch % 10 == 0 and step in draw_list:
                cls_fig = plot_inference(cls_predictions, vertex_coord_list[2], cuda, f"cls{step}_epoch{epoch} Segmentation Predictions" , f"Scans: {test_set[step-1][0]} \nCls Loss: {t_cls_loss} \nCombo Loss: {t_total_loss}\nCls Accuracy: {cls_acc}")
                inst_fig = plot_inference(seg_predictions, vertex_coord_list[2], cuda, f"inst{step}_epoch{epoch} Segmentation Predictions", f"Scans: {1} \nInst Loss: {t_seg_loss} \nCombo Loss: {t_total_loss}\nInst Accuracy: {inst_acc}")
                writer.add_figure(f'test/samples/cls{step}', cls_fig, epoch)
                writer.add_figure(f'test/samples/inst{step}', inst_fig, epoch)

            log(_logfile, f"Test Step {step}... Done")

        ## Tracking the test loss per epoch
        writer.add_scalar('test/loss_per_epoch/', test_loss/last_test_step, epoch+1)
        writer.add_scalar('test/cls_acc_per_epoch/', test_cls_accuracy/last_test_step, epoch+1)
        writer.add_scalar('test/inst_acc_per_epoch/', test_inst_accuracy/last_test_step, epoch+1)

        if test_loss/last_test_step < min_test_loss:
            min_test_loss = test_loss/last_test_step
            min_test_loss_accuracy_cls = test_cls_accuracy/last_test_step 
            min_test_loss_accuracy_inst = test_inst_accuracy/last_test_step
            # min_test_loss_model_param = model.parameters()
            min_test_loss_model_param = model.state_dict()
            min_test_loss_epoch = epoch
            log(_logfile, f"{min_test_loss_epoch} :: new min_test_loss")

        ## Saving the model parameters periodically 
        if epoch % 10 == 9:
        # if epoch % 10 == 0:
            timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
            chkpt_path =  model_directory+"params_epoch"+str(epoch)+"_"+timestamp+".pt"
            torch.save(model.state_dict(), chkpt_path)

            log(_logfile, f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
            log(_logfile, f"[{timestamp}]:  - epoch = {epoch}")
            log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
            log(_logfile, f"[{timestamp}]:  - test_inst_accuracy = {test_inst_accuracy/last_test_step}")
            log(_logfile, f"[{timestamp}]:  - test_cls_accuracy = {test_cls_accuracy/last_test_step}")
            log(_logfile)    

        log(_logfile, f"Epoch {epoch}... Done")


    ## Saving the model parameters of the minimized test loss and the last epoch, including logging
    timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S') 
    min_path = model_directory+"params_epoch"+str(min_test_loss_epoch)+"_for_min_test_loss.pt"
    chkpt_path =  model_directory+"params_epoch"+str(epoch)+"_"+timestamp+".pt"

    torch.save(model.state_dict(), chkpt_path)
    log(_logfile, f"[{timestamp}]: Checkpoint saved - {chkpt_path}")
    log(_logfile, f"[{timestamp}]:  - epoch = {epoch}")
    log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss/last_test_step}")
    log(_logfile, f"[{timestamp}]:  - test_cls_accuracy = {test_cls_accuracy/last_test_step}")
    log(_logfile, f"[{timestamp}]:  - test_inst_accuracy = {test_inst_accuracy/last_test_step}")
    log(_logfile)

    torch.save(min_test_loss_model_param, min_path)
    log(_logfile, f"[{timestamp}]: Saved min_test_loss_model_params - {min_path}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss = {min_test_loss}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss_epoch = {min_test_loss_epoch}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss_accuracy_cls = {min_test_loss_accuracy_cls}")
    log(_logfile, f"[{timestamp}]:  - min_test_loss_accuracy_inst = {min_test_loss_accuracy_inst}")
    log(_logfile)

    stoptime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log(_logfile, f"Stop time: {stoptime}")
    log(_logfile)
    log(_logfile, "Closing logfile")
    _logfile.close()




def train_loc_model(
    nametag, model, train_data_dir, test_data_dir, model_params_path=None,
    batch_size=1, learning_rate=1e-4, reg_constant= 10, epochs=150, 
    MSE_centroid_coeff=1, MSE_label_coeff=1, MSE_poolpt_coeff=1,
    draw_list=[0,1,2,3,4], draw_frequency=20, save_frequency=10,
    cuda=True, debug=False, track_losses=True, testing_sample_size=None,
    scaling="zero-at-min--no-scaling", output_scaling=None, loss_version=1, 
    log_root="./_loc_log/", model_root="./_loc_model/", tensorboard_root="./_loc_runs/"):

    ## Printing Function Arguments
    if debug:  
        print("###################")
        print("Runtime Arguments")
        print("###################")
        print(f"#### nametag: {nametag}")
        # print(f"model: {model}")
        print(f"#### model_params_path: {model_params_path}")

        print(f"#### train_data_dir: {train_data_dir}")
        print(f"#### test_data_dir: {test_data_dir}")
        
        print(f"#### batch_size: {batch_size}")
        print(f"#### learning_rate: {learning_rate}")
        print(f"#### regularization_constant: {reg_constant}")
        print(f"#### epochs: {epochs}")
        print(f"#### cuda: {cuda}")

        print(f"#### draw_list: {draw_list}")
        print(f"#### draw_frequency: {draw_frequency}")
        print(f"#### save_freqency: {save_frequency}")

        print(f"#### log_root: {log_root}")
        print(f"#### model_root: {model_root}")
        print(f"#### tensorboard_root: {tensorboard_root}")
        print("###################")
        print()

    ## For Tagging subdirectory and reporting
    starttime = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

    ## Prepare/Verify Log Directory Locations and Files, but only if desired
    if log_root:
        _logfile_dir =  f"{log_root}{nametag}/"
        if not os.path.isdir(_logfile_dir):
            os.makedirs(_logfile_dir)
        _logfile_path = f"{_logfile_dir}{starttime}.txt"
        _logfile = open(_logfile_path, "w")
    else:
        _logfile = None

    if model_root and save_frequency:
        _model_dir = f"{model_root}{nametag}/"
        if not os.path.isdir(_model_dir):
            os.makedirs(_model_dir)
        
    if tensorboard_root and track_losses:
        _tnsboard_dir = f"{tensorboard_root}{nametag}/"
        if not os.path.isdir(_tnsboard_dir):
            os.makedirs(_tnsboard_dir)

    ## Model-, Data-, Objective- Setup and Header Logging
    if True:
        
        log(_logfile, f"Start Time: {starttime}")


        ## - GPU or CPU
        log(_logfile, f"Cuda available: {torch.cuda.is_available()}")
        log(_logfile, f"Cuda: {cuda}")
        device = (
            "cuda"
            if torch.cuda.is_available() and cuda
            else "cpu"
        )
        log(_logfile, f"device: {device}")

        ## - Model Loading
        if model_params_path:
            log(_logfile, f"Loading previously saved model parameters from {model_params_path} ...")
            model.load_state_dict(torch.load(model_params_path))
            log(_logfile, "Loaded previously saved model parameters.")   

        log(_logfile)
        log(_logfile, "Model:")
        log(_logfile, model)

        ## - Connecting to and Loading from Data and Label Directories
        log(_logfile)
        log(_logfile, f"Verifying training directory location, {train_data_dir}: {os.path.isdir(train_data_dir)}")
        log(_logfile, f"Checking training directory size: {len(os.listdir(train_data_dir))}")

        log(_logfile, f"Verifying test directory location, {test_data_dir}: {os.path.isdir(test_data_dir)}")
        log(_logfile, f"Checking test directory size: {len(os.listdir(test_data_dir))}")
        log(_logfile)
        
        assert os.path.isdir(train_data_dir) , "Specified Training Data Directory does not exist."
        assert os.path.isdir(test_data_dir) , "Specified Test Data Directory does not exist."
        
        train_dataloader, test_dataloader = setupDataLoaders(train_data_dir, test_data_dir, logfile=_logfile, scaling=scaling )

        ## - Loss Function and Optimizer
        loss_fn, optimizer = setupLocRegOptimizer(model,loss_version=loss_version, learning_rate=learning_rate, reg_constant=reg_constant, logfile=_logfile)
    
    ## Running Epochs of training and testing 
    if True:
        training_start = datetime.now()
        training_start_str = datetime.strftime(training_start, '%Y_%m_%d_%H_%M_%S')
        log(_logfile, message=f"Started Training: {training_start_str}")

        ## The run-specific model-params subdir, tensorboard subdir
        model_directory = f"{_model_dir}{training_start_str}/"
        os.makedirs(model_directory)
        writer = SummaryWriter(f"{_tnsboard_dir}{training_start_str}")

        ## To save the minimum-test-loss model parameters
        min_test_loss_model_params = None
        min_test_loss = torch.inf
        min_test_loss_epoch = None
        min_test_loss_training_loss = 0
        


        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            log(_logfile, message=f"Epoch {t+1}\n-------------------------------")
            train_loss, nominal_loss = train(train_dataloader, model, loss_fn, optimizer, 
                                        MSE_centroid_coeff, MSE_label_coeff, MSE_poolpt_coeff, 
                                        logfile=_logfile, testing_sample_size=testing_sample_size, 
                                        output_scaling=output_scaling,
                                        loss_version=loss_version)
            # train_loss = batch_train(train_dataloader, 20, model, loss_fn, optimizer, logfile=_logfile)
            writer.add_scalar('Loss/train/', train_loss, t)
            writer.add_scalar('NominalLoss/train/', nominal_loss, t)
            
            test_loss = 0
            if test_dataloader:    
                if t % draw_frequency == 0: 
                    test_loss, test_nominal_loss, plots = test(test_dataloader, model, loss_fn, logfile=_logfile, epoch=t, return_plots=True, draw_list_by_idx=draw_list, testing_sample_size=testing_sample_size,
                                                    MSE_label_coeff=MSE_label_coeff,
                                                    MSE_centroid_coeff=MSE_centroid_coeff,
                                                    MSE_poolpt_coeff=MSE_poolpt_coeff,
                                                    loss_version=loss_version,
                                                    output_scaling=output_scaling)
                    for draw_idx, plot in enumerate(plots):
                        writer.add_figure(f"test/samples/{draw_list[draw_idx]}", plot, t)
                else:
                    # test_loss = test(test_dataloader, model, loss_fn, logfile=_logfile)                                
                    test_loss = test(test_dataloader, model, loss_fn, logfile=_logfile, epoch=t, return_plots=False, draw_list_by_idx=draw_list, testing_sample_size=testing_sample_size,
                                                    MSE_label_coeff=MSE_label_coeff,
                                                    MSE_centroid_coeff=MSE_centroid_coeff,
                                                    MSE_poolpt_coeff=MSE_poolpt_coeff,
                                                    loss_version=loss_version,
                                                    output_scaling=output_scaling)
                writer.add_scalar('Loss/test/', train_loss, t)
                writer.add_scalar('NominalLoss/test/', test_nominal_loss, t)

                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    min_test_loss_epoch = t
                    min_test_loss_model_params = model.state_dict()
                    min_test_loss_training_loss = train_loss
                
                
            ## Save Model params periodically ie checkpoint
            if t % save_frequency == save_frequency-1:
                timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
                chkpt_path =  model_directory+f"params_epoch_{t}_{timestamp}.pt"
                torch.save(model.state_dict(), chkpt_path)

                log(_logfile, f"[{timestamp}]: Checkpoint model params saved - {chkpt_path}")
                log(_logfile, f"[{timestamp}]:  - epoch = {t}")
                log(_logfile, f"[{timestamp}]:  - train_loss = {train_loss}") 
                if test_loss:
                    log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss}") 
                

        ## Save Last epoch Model params
        timestamp = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
        chkpt_path =  model_directory+f"params_epoch_{t}_{timestamp}.pt"
        torch.save(model.state_dict(), chkpt_path)

        log(_logfile, f"[{timestamp}]: Last epoch model params saved - {chkpt_path}")
        log(_logfile, f"[{timestamp}]:  - epoch = {t}")
        log(_logfile, f"[{timestamp}]:  - train_loss = {train_loss}") 
        if test_loss:
            log(_logfile, f"[{timestamp}]:  - test_loss = {test_loss}") 

        ## Save Best Model Params
        if test_dataloader:
            min_path =  model_directory+f"min_test_loss_params_epoch_{min_test_loss_epoch}.pt"
            torch.save(min_test_loss_model_params, min_path)

            log(_logfile, f"[{timestamp}]: Best model params saved - {min_path}")
            log(_logfile, f"[{timestamp}]:  - epoch = {min_test_loss_epoch}")
            log(_logfile, f"[{timestamp}]:  - train_loss = {min_test_loss_training_loss}") 
            log(_logfile, f"[{timestamp}]:  - test_loss = {min_test_loss}") 

            # model.load_state_dict(min_test_loss_model_params)
            # test_loss, plots = test(test_dataloader, model, loss_fn, logfile=_logfile, return_plots=True, draw_list_by_idx=draw_list)
            # for draw_idx, plot in enumerate(plots):
            #     writer.add_figure(f"test/samples/{draw_list[draw_idx]}", plot, t)


        training_end = datetime.now()
        training_end_str = datetime.strftime(training_end, '%Y_%m_%d_%H_%M_%S')
        log(_logfile, message=f"Finished Training: {training_end_str}")
        
        training_duration = training_end - training_start
        log(_logfile, f"Training Duration: {training_duration}")
        log(_logfile, message="Done!")


