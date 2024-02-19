"""
Goal here is to pacakge training into callable components
So that multiple models can be trained on a single instruction

Possible model variants (each can also be tested on different hyperparams combinations)
 - As is  (relative weighting of the different losses)
 - isolating the semantic segmentation
 - isolating the instance segmentation
 - isolating the locationing regression
 - some skip connections to forward inputs deeper into the chaing
 - inferences earlier in the network, by adding fully connected prediction layers

pass in writer to check diff or look at each separately?
"""

from Instseg_model_prev import MultiLayerFastLocalGraphModelV2 as model0
from Instseg_model import MultiLayerFastLocalGraphModelV2 as model1
from experiment_utils import train_model

##  model_0
"""
Model_0: the original model but adapted for max segmentation instances of 7
Training on the same data: 300 Coupons
Purpose: To see if current train1 run yields is behaving properly on the same training conditions
"""
model_0 = {
    'model': model0(num_classes=3, max_instance_no=7),
    'nametag': "original0",
    'dataset_root': './_data/',
    'batch_size': 1,
    'learning_rate': 1e-4,
    'epochs': 50,
    'initializing_state_dict': None,
    'cuda': True
}

##  model_0
"""
Model_1: the revised model (train1) for max segmentation instances of 7
Training on the same data: 300 Coupons + 
"""
model_1 = {
    'model': model1(num_classes=3, max_instance_no=7),
    'nametag': "train1-fix2",
    'dataset_root': './_data/',
    'batch_size': 1,
    'learning_rate': 1e-4,
    'epochs': 150,
    'initializing_state_dict': './_model/train1-fix2/2023_06_15_09_03_24/params_epoch97_for_min_test_loss.pt',
    'cuda': True,
    'train_data_subset': None,
    'test_data_subset': None,
    'draw_list': [0,1,2,3,4, 90, 100, 110]
}

## Train1-fix3
"""
Starting from Same Model_1 trained on coupons and floorpan.bins
Changing the data set to the inverted .plys
Required a different scans/, /labels/, test.txt, and train.txt
"""
# train1_fix3 = {
#     'model': model1(num_classes=3, max_instance_no=7),
#     'nametag': "train1-fix3",
#     'dataset_root': './_data/',
#     'batch_size': 1,
#     'learning_rate': 1e-4,
#     'epochs': 500,
#     'initializing_state_dict': './_model/train1-fix2/2023_06_16_09_11_24/params_epoch142_for_min_test_loss.pt',
#     'cuda': True,
#     'train_data_subset': None,
#     'test_data_subset': None,
#     'draw_list': [0,1,2,3,4, 90, 100, 110] 
# }

# train1_fix3 = {
#     'model': model1(num_classes=3, max_instance_no=7),
#     'nametag': "train1-fix3",
#     'dataset_root': './_data/',
#     'batch_size': 1,
#     'learning_rate': 1e-4,
#     'epochs': 500,
#     'initializing_state_dict': './_model/train1-fix3/2023_06_22_17_01_13/params_epoch497_for_min_test_loss.pt',
#     'cuda': True,
#     'train_data_subset': None,
#     'test_data_subset': None,
#     'draw_list': [0,1,2,3,4, 90, 100, 110, 102, 25, 77, 95, 11, 36, 59] 
# }

# train1_fix3 = {
#     'model': model1(num_classes=3, max_instance_no=7),
#     'nametag': "train1-fix3",
#     'dataset_root': './_data/',
#     'batch_size': 1,
#     'learning_rate': 1e-4,
#     'epochs': 500,
#     'initializing_state_dict': './_model/train1-fix3/2023_06_30_09_56_12/params_epoch488_for_min_test_loss.pt',
#     'cuda': True,
#     'train_data_subset': None,
#     'test_data_subset': None,
#     'draw_list': [0,1,2,3,4, 90, 100, 110, 102, 25, 77, 95, 11, 36, 59] 
# }

train1_fix3 = {
    'model': model1(num_classes=3, max_instance_no=7),
    'nametag': "train1-fix3",
    'dataset_root': './_data/',
    'batch_size': 1,
    'learning_rate': 1e-4,
    'epochs': 2000,
    'initializing_state_dict': './_model/train1-fix3/2023_07_06_12_07_23/params_epoch498_for_min_test_loss.pt',
    'cuda': True,
    'train_data_subset': None,
    'test_data_subset': None,
    'draw_list': [0,1,2,3,4, 90, 100, 110, 102, 25, 77, 95, 11, 36, 59] 
}

# train1_fix3_temp = {
#     'model': model1(num_classes=3, max_instance_no=7),
#     'nametag': "train1-fix3_temp",
#     'dataset_root': './_data/',
#     'batch_size': 1,
#     'learning_rate': 1e-4,
#     'epochs': 2,
#     'initializing_state_dict': './_model/train1-fix3/2023_06_30_09_56_12/params_epoch488_for_min_test_loss.pt',
#     'cuda': True,
#     'train_data_subset': None,
#     'test_data_subset': None,
#     'draw_list': [0,1,2,3,4, 90, 100, 110, 102, 25, 77, 95, 11, 36, 59] 
# }


## RERUN -- from start to 2000 so that the whole training history is captured in one tensorboard log
train_fix3_full2000 = {
    'model': model1(num_classes=3, max_instance_no=7),
    'nametag': "train1-fix3_full2000",
    'dataset_root': './_data/',
    'batch_size': 1,
    'learning_rate': 1e-4,
    'epochs': 2000,
    'initializing_state_dict': None,
    'cuda': True,
    'train_data_subset': None,
    'test_data_subset': None,
    'draw_list': [0,1,2,3,4, 90, 100, 110, 102, 25, 77, 95, 11, 36, 59] 
}

# models = [train1_fix3_temp]
# models = [train1_fix3]
# models = [train_fix3_full2000]
models = [model_0]

for model in models:
    train_model(**model)


# current terminal pid 
# 3276765.pts-2.USMCDLDCN273KD4   (11/04/2023 02:04:29 PM)        (Detached)
