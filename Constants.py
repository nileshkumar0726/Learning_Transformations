device_T = 'gpu'
tess_size = [20,20]

exp_type = 'image_input'
IMG_FOLDER = "/home/nk4856/data/Datasets/Cardiac_Dataset/LV_Dataset/LV_Labels"
logs_folder = "logs"


epochs = 400
batch_size = 16
lr = 0.001 
weight_decay = 1e-5
img_dimensions = (200,200)
regularization_constant = 1

total_train_samples = 500
total_val_samples = 100

configuration = 'Exp type {exp_type}, batch_size {batch_size}, learning_rate {lr}, velocity_reg_weight {rg_const} '\
    .format(exp_type=exp_type, batch_size=batch_size, lr=lr, rg_const=regularization_constant)