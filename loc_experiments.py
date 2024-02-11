from LocReg_model import LocationRegressionModelV1 as model1
from experiment_utils import train_loc_model

data_root = "./Recreating Dataset/locationing_dataset/"
train_data_dir = data_root+"train/"
test_data_dir = data_root+"test/"


expt1 = {
    'model': model1(),
    'nametag': "original_test",
    'train_data_dir': train_data_dir, 
    'test_data_dir': test_data_dir, 
    'log_root': "./_loc_log/",
    'model_root': "./_loc_model/", 
    'tensorboard_root': "./_loc_runs/" ,
    'batch_size': 1, 
    'learning_rate': 1e-4, 
    'reg_constant': 0,
    'epochs': 300, 
    'draw_list': [0,2,23, 40, 10], 
    'cuda': True, 
    'model_params_path': None, 
    'debug': False,
    'draw_frequency':1, 
    'save_frequency':1,
}


experiments = [expt1]
for expt in experiments:
    train_loc_model(**expt)