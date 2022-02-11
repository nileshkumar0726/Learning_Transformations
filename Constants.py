


device_T = 'gpu'
tess_size = [4,4]

isTumor = True
normalize = True 
exp_type = 'Tumor'
IMG_FOLDER = "/home/nk4856/data/Datasets/Liver_Dataset/new_data/train/sliced/seg"
#"/home/nk4856/data/Datasets/Cardiac_Dataset/LV_Dataset/LV_Labels"
TUMOR_SEPERATED_FOLDER = "/home/nk4856/data/Datasets/Liver_Dataset/Extracted_Tumor_Labels_Remove_Consecutive" #this folder contains tumors 

max_slice_no = 362
logs_folder = "logs"


epochs = 400
batch_size = 10
lr = 0.0001 
weight_decay = 1e-5
img_dimensions = (200,200)
regularization_constant = 0.001
Checkpoint_folder = "Model_Weights"


total_train_samples = 2200
total_val_samples = 100

configuration = 'Exp_type_{exp_type}_batch_size_{batch_size}_learning_rate_{lr}_velocity_reg_weight_{rg_const}_'\
    .format(exp_type=exp_type, batch_size=batch_size, lr=lr, rg_const=regularization_constant)


MAX_HEIGHT = 73
MAX_WIDTH = 86