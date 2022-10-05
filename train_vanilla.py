import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from Models.Vanilla_VAE import Vanilla_VAE
from Utils.util import UtilityFunctions
from Constants import total_train_samples, total_val_samples, \
    batch_size, epochs, lr, weight_decay, img_dimensions, logs_folder, configuration, isTumor, \
        normalize, is_determinstic, max_patience, velocity_lambda
from Datasets.pairs_dataset import PairsDataset
from torch.utils.data import DataLoader
import datetime
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def train_vae ():

    
    train_imgs, train_labels, train_paths =\
         UtilityFunctions.load_tumor_samples (start=0, end=total_train_samples, normalize=normalize, size=img_dimensions)

    train_pairs = UtilityFunctions.make_pairs_list_modified_KNN (train_imgs, train_labels)
    train_dataset = PairsDataset (train_pairs, train_imgs, train_paths, isTumor=isTumor)
    train_loader = DataLoader (train_dataset, shuffle = True, batch_size = batch_size, drop_last=True)

    val_imgs, val_labels, val_paths = UtilityFunctions.load_tumor_samples (start=total_train_samples, \
        end=total_train_samples + total_val_samples, normalize=normalize, size=img_dimensions)

    val_pairs = UtilityFunctions.make_pairs_list_modified_KNN (val_imgs, val_labels)
    val_dataset = PairsDataset (val_pairs, val_imgs, val_paths, isTumor=isTumor)
    val_loader = DataLoader (val_dataset, shuffle = False, batch_size = 8, drop_last=True)

    model = Vanilla_VAE ()

    fit (model, train_loader, val_loader)




def fit (model, train_loader, val_loader):

    min_val_loss = 10000
    curr_patience = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam (model.parameters(), lr = lr, weight_decay = weight_decay)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    
    model = model.to(device)

    log_folder =  os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter (log_folder, comment=configuration)


    for i in range (epochs):

        print ("Epoch = ", i)

        running_recon_loss = 0.0
        running_kld_loss = 0.0
        running_velocity_norm = 0.0

        running_val_recon_loss = 0.0
        running_val_kld_loss = 0.0
        running_val_velocity_norm = 0.0

        model.train()
        iter = 0

        for src, tgt, _ in tqdm(train_loader):

            src = src.to(device).float()
            tgt = tgt.to(device).float()

            x = torch.cat((src, tgt), dim = 1)
            
            optimizer.zero_grad()

            if is_determinstic:

                reconstruction = model (x)
                loss_matrix = torch.norm (reconstruction - tgt)
                BCE_loss = torch.norm (loss_matrix)
                loss = BCE_loss
                KLD = torch.tensor(0)
                

            else:

                reconstruction, mu, logvar, z, velocities = model(x, return_velocites= True)
                loss_matrix = torch.norm (reconstruction - tgt)
                BCE_loss = torch.norm (loss_matrix)
                velocity_norm = torch.norm(velocities) * velocity_lambda
                BCE_loss, KLD = UtilityFunctions.final_loss(BCE_loss, mu, logvar)
                loss = BCE_loss + KLD + velocity_norm
                
            loss.backward()
            optimizer.step()
            
            assert (reconstruction.size() == tgt.size())
            
            writer.add_scalar ('Loss/train_velocity_norm', velocity_norm.item()/batch_size, i*len(train_loader.dataset) + iter)
            writer.add_scalar ('Loss/train_recon',BCE_loss.item()/batch_size, i*len(train_loader.dataset) + iter )
            writer.add_scalar ('Loss/train_kld',KLD.item()/batch_size, i*len(train_loader.dataset) + iter )
            
            iter += 1
        
            

            running_recon_loss += BCE_loss.item()
            running_kld_loss += KLD.item()
            running_velocity_norm += velocity_norm.item()
            

        src_grid = make_grid(src)
        tgt_grid = make_grid(tgt)
        recon_grid = make_grid(reconstruction)
        

        writer.add_image ('Images/Src',src_grid, i)
        writer.add_image ('Images/Tgt',tgt_grid, i)
        writer.add_image ('Images/Recon',recon_grid, i)


        # running_recon_loss /= len(train_loader.dataset)
        # running_kld_loss /= len(train_loader.dataset)
        # running_velocity_norm /= len(train_loader.dataset)
            
        # src_grid = make_grid(src)
        # tgt_grid = make_grid(tgt)
        # recon_grid = make_grid(reconstruction)
        # src_img_grid = make_grid (src_img)
        # recon_src_image_grid = make_grid (reconstruction_img)
        

        # writer.add_image ('Images/Src',src_grid, i)
        # writer.add_image ('Images/Tgt',tgt_grid, i)
        # writer.add_image ('Images/Recon',recon_grid, i)
        # writer.add_image ('Images/Recon_Src_Img',recon_src_image_grid, i)
        # writer.add_image ('Images/Src_Img',src_img_grid, i)


        #Val Loop - Train and Val need to be coded as a single func with flags to avoid repeating code
        model.eval()

        for src, tgt, _ in tqdm(val_loader):

            src = src.to(device).float()
            tgt = tgt.to(device).float()

            x = torch.cat((src, tgt), dim = 1)
            
            if is_determinstic:

                reconstruction = model (x)
                loss_matrix = torch.norm (reconstruction - tgt)
                BCE_loss = torch.norm (loss_matrix)
                loss = BCE_loss
                KLD = torch.tensor(0)
                
            else:

                reconstruction, mu, logvar, z, velocities = model(x, return_velocites = True)
                loss_matrix = torch.norm (reconstruction - tgt)
                BCE_loss = torch.norm (loss_matrix)
                velocity_norm = torch.norm(velocities) * velocity_lambda
                BCE_loss, KLD = UtilityFunctions.final_loss(BCE_loss, mu, logvar)
                loss = BCE_loss + KLD + velocity_norm

            assert (reconstruction.size() == tgt.size())


            running_val_recon_loss += BCE_loss.item()
            running_val_kld_loss += KLD.item()
            running_val_velocity_norm += velocity_norm.item()

        running_val_recon_loss /= len(val_loader.dataset)
        running_val_kld_loss /= len(val_loader.dataset)
        running_val_velocity_norm /= len (val_loader.dataset)
        
        writer.add_scalar ('Loss/val_recon',running_val_recon_loss, i )
        writer.add_scalar ('Loss/val_kld',running_val_kld_loss, i )
        writer.add_scalar ('Loss/val_velocity_norm', running_val_velocity_norm, i)

        src_grid = make_grid(src)
        tgt_grid = make_grid(tgt)
        recon_grid = make_grid(reconstruction)

        writer.add_image ('Val_Images/Src',src_grid, i)
        writer.add_image ('Val_Images/Tgt',tgt_grid, i)
        writer.add_image ('Val_Images/Recon',recon_grid, i)
        
        total_val_loss = running_val_recon_loss + running_val_kld_loss + running_val_velocity_norm

        if total_val_loss < min_val_loss:

            print ("Model saved in " + str(i) +"th epoch")
            UtilityFunctions.save_checkpoint (i, model, optimizer, running_recon_loss)
            min_val_loss = total_val_loss
            curr_patience = 0
        
        else:
            if curr_patience == max_patience:
                print ("Max patience reached, finis")












