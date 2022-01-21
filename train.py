import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
#from Models.Region_Model import Region_Specific_VAE
from Models.Two_Transformation_Model import Region_Specific_VAE
from Utils.util import UtilityFunctions
from Constants import total_train_samples, total_val_samples, \
    batch_size, epochs, lr, weight_decay, regularization_constant, logs_folder, configuration, isTumor, \
        normalize
from Datasets.pairs_dataset import PairsDataset
from torch.utils.data import DataLoader
import datetime
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def train_vae ():

    model = Region_Specific_VAE ()
    
    train_imgs, train_labels, train_paths = UtilityFunctions.load_tumor_samples (start=0, end=total_train_samples, normalize=normalize)
    train_pairs = UtilityFunctions.make_pairs_list_modified_KNN (train_imgs, train_labels)
    train_dataset = PairsDataset (train_pairs, train_imgs, train_paths, isTumor=isTumor)
    train_loader = DataLoader (train_dataset, shuffle = True, batch_size = batch_size, drop_last=True)

    val_imgs, val_labels, val_paths = UtilityFunctions.load_tumor_samples (start=total_train_samples, \
        end=total_train_samples + total_val_samples, normalize=normalize)

    val_pairs = UtilityFunctions.make_pairs_list_modified_KNN (val_imgs, val_labels)
    val_dataset = PairsDataset (val_pairs, val_imgs, val_paths, isTumor=isTumor)
    val_loader = DataLoader (val_dataset, shuffle = False, batch_size = 4, drop_last=True)

    fit (model, train_loader, val_loader)




def fit (model, train_loader, val_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam (model.parameters(), lr = lr, weight_decay = weight_decay)
    model = model.to(device)

    log_folder =  os.path.join(logs_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter (log_folder, comment=configuration)


    for i in range (epochs):

        running_recon_loss = 0.0
        running_kld_loss = 0.0
        running_reg_loss = 0.0

        running_val_recon_loss = 0.0
        running_val_kld_loss = 0.0
        running_val_reg_loss = 0.0

        model.train()

        for src, tgt, src_img in tqdm(train_loader):

            src = src.to(device).float()
            tgt = tgt.to(device).float()
            src_img = src_img.to(device).float()

            x = torch.cat((src, tgt), dim = 1)
            
            optimizer.zero_grad()
            reconstruction, mu, logvar, z, velocities, reconstruction_img = model(x, src_img=src_img)

            src_bboxes = UtilityFunctions.extract_bbox (src.detach().cpu().numpy())
            tgt_bboxes = UtilityFunctions.extract_bbox (tgt.detach().cpu().numpy())

            it = 0
            while it < len(src_bboxes):

                x_n_bbox = src_bboxes[it]
                x_m_bbox = tgt_bboxes[it]

                if isTumor:

                    x_n_bbox, x_m_bbox = UtilityFunctions.match_bboxes (x_n_bbox, x_m_bbox)
                    loss_matrix = UtilityFunctions.augmented_distance(reconstruction[it] , tgt[it] ,x_n_bbox, x_m_bbox)


                else:

                    bbox = UtilityFunctions.union_bboxes (x_n_bbox, x_m_bbox)
                    diff_matrix = tgt[it] - reconstruction[it]
                    loss_matrix = diff_matrix[:,bbox[0]:bbox[2], bbox[1]:bbox[3]]
                
                
                if it == 0:
                    bce_loss = torch.norm(loss_matrix)
                else:
                    bce_loss += torch.norm(loss_matrix)
                it += 1

            BCE_loss, KLD = UtilityFunctions.final_loss(bce_loss, mu, logvar)
            loss = BCE_loss + KLD 

            velocity_regularization = torch.norm (velocities)
            velocity_regularization = regularization_constant * velocity_regularization
            loss = loss + velocity_regularization

            loss.backward()
            optimizer.step()

            running_recon_loss += BCE_loss.item()
            running_kld_loss += KLD.item()
            running_reg_loss += velocity_regularization.item()

        running_recon_loss /= len(train_loader.dataset)
        running_kld_loss /= len(train_loader.dataset)
        running_reg_loss /= len(train_loader.dataset)
        
        writer.add_scalar ('Loss/train_recon',running_recon_loss, i )
        writer.add_scalar ('Loss/train_kld',running_kld_loss, i )
        writer.add_scalar ('Loss/train_reg',running_reg_loss, i )

        src_grid = make_grid(src)
        tgt_grid = make_grid(tgt)
        recon_grid = make_grid(reconstruction)
        src_img_grid = make_grid (src_img)
        recon_src_image_grid = make_grid (reconstruction_img)
        

        writer.add_image ('Images/Src',src_grid, i)
        writer.add_image ('Images/Tgt',tgt_grid, i)
        writer.add_image ('Images/Recon',recon_grid, i)
        writer.add_image ('Images/Recon_Src_Img',recon_src_image_grid, i)
        writer.add_image ('Images/Src_Img',src_img_grid, i)


        #Val Loop
        model.eval()

        for src, tgt, src_img in tqdm(val_loader):

            src = src.to(device).float()
            tgt = tgt.to(device).float()
            src_img = src_img.to(device).float()

            x = torch.cat((src, tgt), dim = 1)
            
            reconstruction, mu, logvar, z, velocities, reconstruction_img = model(x, src_img=src_img)

            src_bboxes = UtilityFunctions.extract_bbox (src.detach().cpu().numpy())
            tgt_bboxes = UtilityFunctions.extract_bbox (tgt.detach().cpu().numpy())

            it = 0
            while it < len(src_bboxes):

                x_n_bbox = src_bboxes[it]
                x_m_bbox = tgt_bboxes[it]

                bbox = np.zeros_like (x_m_bbox)
                
                bbox[0] = min (x_n_bbox[0], x_m_bbox[0])
                bbox[1] = min (x_n_bbox[1], x_m_bbox[1])
                bbox[2] = max (x_n_bbox[2], x_m_bbox[2])
                bbox[3] = max (x_n_bbox[3], x_m_bbox[3])

                if it == 0:
                    plot_box = bbox

                diff_matrix = tgt[it] - reconstruction[it]
                loss_matrix = diff_matrix[:,bbox[0]:bbox[2], bbox[1]:bbox[3]]
                if it == 0:
                    bce_loss = torch.norm(loss_matrix)
                else:
                    bce_loss += torch.norm(loss_matrix)
                it += 1

            BCE_loss, KLD = UtilityFunctions.final_loss(bce_loss, mu, logvar)
            loss = BCE_loss + KLD 

            velocity_regularization = torch.norm (velocities)
            velocity_regularization = regularization_constant * velocity_regularization
            loss = loss + velocity_regularization


            running_val_recon_loss += BCE_loss.item()
            running_val_kld_loss += KLD.item()
            running_val_reg_loss += velocity_regularization.item()

        running_val_recon_loss /= len(val_loader.dataset)
        running_val_kld_loss /= len(val_loader.dataset)
        running_val_reg_loss /= len(val_loader.dataset)
        
        writer.add_scalar ('Loss/val_recon',running_val_recon_loss, i )
        writer.add_scalar ('Loss/val_kld',running_val_kld_loss, i )
        writer.add_scalar ('Loss/val_reg',running_val_reg_loss, i )

        src_grid = make_grid(src)
        tgt_grid = make_grid(tgt)
        recon_grid = make_grid(reconstruction)
        src_img_grid = make_grid (src_img)
        recon_src_image_grid = make_grid (reconstruction_img)


        writer.add_image ('Val_Images/Src',src_grid, i)
        writer.add_image ('Val_Images/Tgt',tgt_grid, i)
        writer.add_image ('Val_Images/Recon',recon_grid, i)
        writer.add_image ('Val_Images/Recon_Src_Img',recon_src_image_grid, i)
        writer.add_image ('Val_Images/Src_Img',src_img_grid, i)












