import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from Models.Region_Model import Region_Specific_VAE
from Utils.util import UtilityFunctions
from Constants import total_train_samples, total_val_samples, \
    batch_size, epochs, lr, weight_decay, regularization_constant, logs_folder, configuration
from Datasets.pairs_dataset import PairsDataset
from torch.utils.data import DataLoader
import datetime
import os
from torch.utils.tensorboard import SummaryWriter


def train_vae ():

    model = Region_Specific_VAE ()
    
    train_imgs, train_labels = UtilityFunctions.load_samples (start=0, end=total_train_samples)
    train_pairs = UtilityFunctions.make_pairs_list_KNN (train_imgs, train_labels)
    train_dataset = PairsDataset (train_pairs, train_imgs)
    train_loader = DataLoader (train_dataset, shuffle = True, batch_size = batch_size)

    val_imgs, val_labels = UtilityFunctions.load_samples (start=total_train_samples, \
        end=total_train_samples + total_val_samples)

    val_pairs = UtilityFunctions.make_pairs_list_KNN (val_imgs, val_labels)
    val_dataset = PairsDataset (val_pairs, val_imgs)
    val_loader = DataLoader (val_dataset, shuffle = True, batch_size = batch_size)

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

        for src, tgt in train_loader:

            src = src.to(device).float()
            tgt = tgt.to(device).float()

            x = torch.cat((src, tgt), dim = 1)
            
            optimizer.zero_grad()
            reconstruction, mu, logvar, z, velocities = model(x)

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










