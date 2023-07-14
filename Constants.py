


device_T = 'gpu'
tess_size = [20,20]


is_determinstic = False
isTumor = True
normalize = True 
exp_type = 'Tumor'
IMG_FOLDER = '/home/stu3/s15/nk4856/Research/Pytorch-UNet/Data/new_data/train/sliced/ct'
#'/home/stu3/s15/nk4856/Research/LV_Dataset/LV_Labels'
#"/home/stu3/s15/nk4856/Research/Pytorch-UNet/Data/new_data/train/sliced/seg"
TUMOR_SEPERATED_FOLDER = "/home/stu3/s15/nk4856/Research/KiTS_Tumors"
#"/home/stu3/s15/nk4856/Research/Pytorch-UNet/Data/Extracted_Tumor_Labels_Remove_Consecutive" #this folder contains tumors 

max_slice_no = 362
logs_folder = "logs"




epochs = 400
batch_size = 16
lr = 0.0001 
max_patience = 20
latent_dim = 12
weight_decay = 1e-5
img_dimensions = (30,30)
regularization_constant = 2 #0.5 for tumor
Checkpoint_folder = "Model_Weights"
velocity_lambda = 0.001 #to control effect of velocity_magnitude on overall loss
beta = 0.001

total_train_samples =  1500 #2200 for tumor
total_val_samples = 200

configuration = 'Exp_type_{exp_type}_batch_size_{batch_size}_learning_rate_{lr}_velocity_reg_weight_{rg_const}_'\
    .format(exp_type=exp_type, batch_size=batch_size, lr=lr, rg_const=regularization_constant)


MAX_HEIGHT = 73
MAX_WIDTH = 86


fold_4_50_percent_train = ['00000', '00003', '00005', '00006', '00008', '00010', '00013', '00015', '00016', '00020', '00021', '00025', '00027', '00028', '00033', '00035', '00037', '00040', '00041', '00042', '00044', '00045', '00046', '00047', '00048', '00049', '00050', '00052', '00055', '00056', '00060', '00062', '00063', '00070', '00074', '00078', '00079', '00086', '00088', '00090', '00093', '00094', '00095', '00100', '00102', '00107', '00110', '00111', '00116', '00125', '00127', '00132', '00135', '00137', '00141', '00146', '00147', '00151', '00154', '00159', '00160', '00161', '00162', '00163', '00165', '00167', '00168', '00171', '00173', '00174', '00177', '00179', '00180', '00181', '00182', '00184', '00185', '00187', '00191', '00192', '00194', '00205', '00206', '00208']
fold_3_50_percent_train = ['00004', '00007', '00016', '00017', '00019', '00024', '00026', '00027', '00028', '00031', '00033', '00035', '00039', '00040', '00042', '00044', '00047', '00049', '00051', '00056', '00059', '00061', '00064', '00067', '00068', '00070', '00073', '00077', '00078', '00081', '00084', '00086', '00089', '00090', '00093', '00095', '00098', '00100', '00103', '00104', '00105', '00106', '00109', '00113', '00114', '00115', '00118', '00119', '00120', '00121', '00124', '00125', '00126', '00133', '00135', '00136', '00139', '00140', '00143', '00148', '00149', '00152', '00154', '00155', '00157', '00160', '00162', '00163', '00164', '00165', '00171', '00173', '00180', '00181', '00184', '00189', '00192', '00193', '00194', '00196', '00201', '00202', '00204', '00205']