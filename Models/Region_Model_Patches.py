import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab import Cpab
from Constants import tess_size, device_T
from Utils.util import UtilityFunctions



"""
Instead of passing complete images to encoder
just pass patches contatining tumor mask to 
the encoder


Todo: Changie naming convention to make it general
"""

class Region_Specific_Model_Liver_Patches_VAE_DI(nn.Module):
    def __init__(self, img_channels = 2, z_dim = 20):
        super(Region_Specific_Model_Liver_Patches_VAE_DI, self).__init__()


        self.z_dim = z_dim
        self.T = Cpab(tess_size=tess_size, backend='pytorch', device=device_T, zero_boundary=True)
 
        # encoder
        self.conv1 = nn.Conv2d(img_channels, out_channels = 64, kernel_size = 3, dilation=5) #output size 63,76
        self.conv2 = nn.Conv2d(64, out_channels = 64, kernel_size = 3, dilation=5) #output size 53,66
        self.conv3 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size 33,46
        self.conv4 = nn.Conv2d(32, out_channels = 32, kernel_size = 5, dilation=5) #output size 13,26
        # self.conv5 = nn.Conv2d(32, out_channels = 16, kernel_size = 5, dilation=5) #output size 156
        # self.conv6 = nn.Conv2d(16, out_channels = 16, kernel_size = 5, dilation=5) #output size 136
        # self.conv7 = nn.Conv2d(16, out_channels = 8, kernel_size = 5, dilation=5) #output size 116
        # self.conv8 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 96
        # self.conv9 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 76
        # self.conv10 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 56
        # self.conv11 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 36



        self.enc1 = nn.Linear(in_features=13*26*32, out_features=1024)
        self.enc2 = nn.Linear (in_features=1024, out_features = 512 )
        self.enc3 = nn.Linear (in_features=512, out_features = 256 )
        self.enc4 = nn.Linear (in_features=256, out_features = 128 )
        self.enc5 = nn.Linear (in_features=128, out_features = z_dim ) #mu 
        self.enc6 = nn.Linear (in_features=128, out_features = z_dim ) #var


        #Decoder Conv Layers (to process the image input before passing to the decoder)
        self.convD1 = nn.Conv2d(1, out_channels = 64, kernel_size = 5, dilation=5) #output size 180
        self.convD2 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size 160
        self.convD3 = nn.Conv2d(32, out_channels = 16, kernel_size = 5, dilation=5) #output size 140
        self.convD4 = nn.Conv2d(16, out_channels = 16, kernel_size = 5, dilation=5) #output size 120
        self.convD5 = nn.Conv2d(16, out_channels = 8, kernel_size = 5, dilation=5) #output size 100
        self.convD6 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 80
        self.convD7 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 60
        self.convD8 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 40
        self.convD9 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 20


        # decoder full connected layers for theta_1
        self.FCD1 = nn.Linear(in_features=z_dim + (20*20*8), out_features=2400)
        self.FCD2 = nn.Linear(in_features=2400, out_features=2000)
        self.FCD3 = nn.Linear (in_features=2000, out_features = 1800)
        self.FCD4 = nn.Linear (in_features=1800, out_features = 1700)
        self.FCD5 = nn.Linear(in_features=1700, out_features=self.T.params.d)

        #decoder full connected for theta_2
        self.FCD6 = nn.Linear(in_features=z_dim, out_features=256)
        self.FCD7 = nn.Linear(in_features=256, out_features=512)
        self.FCD8 = nn.Linear(in_features=512, out_features=1024)
        self.FCD9 = nn.Linear(in_features=1024, out_features=512)
        self.FCD10 = nn.Linear(in_features=512, out_features=256)

        self.FCD11 = nn.Linear (in_features=256, out_features=self.T.params.d)



    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode (self, x):

        #x is concatenated two images
        x_out = self.conv1 (x)
        x_out = F.leaky_relu(x_out)

        x_out = self.conv2 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv4 (x_out)
        x_out = F.leaky_relu (x_out)


        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.enc1 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.enc3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc4 (x_out)
        x_out = F.leaky_relu (x_out)
        
        mu = self.enc5 (x_out)
        log_var = self.enc6 (x_out)

        z = self.reparameterize (mu, log_var)
        return z, mu, log_var

    def decode (self, x, src_img):

        src_img_out = self.convD1 (src_img)
        src_img_out = F.leaky_relu(src_img_out)

        src_img_out = self.convD2 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD3 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD4 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD5 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD6 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD7 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD8 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD9 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = src_img_out.view(src_img_out.size(0), -1)

        #decoder should spit out two thetas
        x_out = torch.cat((x,src_img_out), dim=1 )
        
        x_out = self.FCD1 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.FCD4 (x_out)
        x_out = F.leaky_relu (x_out)

        theta_1 = self.FCD5 (x_out)

        #second theta
        x = self.FCD6 (x)
        x = F.leaky_relu (x)

        x = self.FCD7 (x)
        x = F.leaky_relu (x)

        x = self.FCD8 (x)
        x = F.leaky_relu (x)

        x = self.FCD9 (x)
        x = F.leaky_relu (x)

        x = self.FCD10 (x)
        x = F.leaky_relu (x)


        theta_2 = self.FCD11 (x)

        return theta_1, theta_2
 
    def forward(self, x, src_img = None, src_mask=None, use_src_mask=False):

        # encoding
        z, mu, log_var = self.encode (x)

        """
        Instead of Passing of exact mask in the decoder
        just pass a rectangle around region of interest
        """
        src_clones = x[:,0,:,:].unsqueeze(1).clone()
        src_mask_bboxes = UtilityFunctions.extract_bbox (src_clones.detach().cpu().numpy())
        
        for idx,bbox in enumerate (src_mask_bboxes):
            src_clones[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1

        if not use_src_mask:    
            # decoding
            theta_1, theta_2 = self.decode (z, src_clones)        
        else:
            theta = self.decode (z, src_mask.unsqueeze(1))
            
        reconstruction, velocities = self.T.transform_data(x[:,0,:,:].unsqueeze(1), theta_1, outsize=x[:,0,:,:].unsqueeze(1).size()[2:], return_velocities=True)
        reconstruction_img = self.T.transform_data(src_img, theta_1, outsize=x[:,0,:,:].unsqueeze(1).size()[2:], return_velocities=False)
        
        ###Do patch wise transformation here
        for idx,bbox in enumerate (src_mask_bboxes):
            src_mask_patch = x[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)
            src_img_patch = src_img[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab import Cpab
from Constants import tess_size, device_T



"""
Instead of passing complete images to encoder
just pass patches contatining tumor mask to 
the encoder


Todo: Changie naming convention to make it general
"""

class Region_Specific_Model_Liver_Patches_VAE_DI(nn.Module):
    def __init__(self, img_channels = 2, z_dim = 35):
        super(Region_Specific_Model_Liver_Patches_VAE_DI, self).__init__()


        self.T = Cpab(tess_size=tess_size, backend='pytorch', device=device_T, zero_boundary=True)
 
        # encoder
        self.conv1 = nn.Conv2d(img_channels, out_channels = 64, kernel_size = 3, dilation=5) #output size 63,76
        self.conv2 = nn.Conv2d(64, out_channels = 64, kernel_size = 3, dilation=5) #output size 53,66
        self.conv3 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size 33,46
        self.conv4 = nn.Conv2d(32, out_channels = 32, kernel_size = 5, dilation=5) #output size 13,26



        self.enc1 = nn.Linear(in_features=13*26*32, out_features=1024)
        self.enc2 = nn.Linear (in_features=1024, out_features = 512 )
        self.enc3 = nn.Linear (in_features=512, out_features = 256 )
        self.enc4 = nn.Linear (in_features=256, out_features = 128 )
        self.enc5 = nn.Linear (in_features=128, out_features = z_dim ) #mu 
        self.enc6 = nn.Linear (in_features=128, out_features = z_dim ) #var


        #Decoder Conv Layers (to process the image input before passing to the decoder)
        self.convD1 = nn.Conv2d(1, out_channels = 64, kernel_size = 5, dilation=5) #output size 180
        self.convD2 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size 160
        self.convD3 = nn.Conv2d(32, out_channels = 16, kernel_size = 5, dilation=5) #output size 140
        self.convD4 = nn.Conv2d(16, out_channels = 16, kernel_size = 5, dilation=5) #output size 120
        self.convD5 = nn.Conv2d(16, out_channels = 8, kernel_size = 5, dilation=5) #output size 100
        self.convD6 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 80
        self.convD7 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 60
        self.convD8 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 40
        self.convD9 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 20


        # decoder full connected layers for theta_1
        self.FCD1 = nn.Linear(in_features=z_dim + (20*20*8), out_features=2400)
        self.FCD2 = nn.Linear(in_features=2400, out_features=2000)
        self.FCD3 = nn.Linear (in_features=2000, out_features = 1800)
        self.FCD4 = nn.Linear (in_features=1800, out_features = 1700)
        self.FCD5 = nn.Linear(in_features=1700, out_features=self.T.params.d)

        #decoder full connected for theta_2
        self.FCD6 = nn.Linear(in_features=z_dim, out_features=256)
        self.FCD7 = nn.Linear(in_features=256, out_features=512)
        self.FCD8 = nn.Linear(in_features=512, out_features=1024)
        self.FCD9 = nn.Linear(in_features=1024, out_features=512)
        self.FCD10 = nn.Linear(in_features=512, out_features=256)

        self.FCD11 = nn.Linear (in_features=256, out_features=self.T.params.d)



    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode (self, x):

        #x is concatenated two images
        x_out = self.conv1 (x)
        x_out = F.leaky_relu(x_out)

        x_out = self.conv2 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv4 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.enc1 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.enc3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc4 (x_out)
        x_out = F.leaky_relu (x_out)
        
        mu = self.enc5 (x_out)
        log_var = self.enc6 (x_out)

        z = self.reparameterize (mu, log_var)
        return z, mu, log_var

    def decode (self, x, src_img):

        src_img_out = self.convD1 (src_img)
        src_img_out = F.leaky_relu(src_img_out)

        src_img_out = self.convD2 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD3 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD4 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD5 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD6 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD7 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD8 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD9 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = src_img_out.view(src_img_out.size(0), -1)

        #decoder should spit out two thetas
        x_out = torch.cat((x,src_img_out), dim=1 )
        
        x_out = self.FCD1 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.FCD4 (x_out)
        x_out = F.leaky_relu (x_out)

        theta_1 = self.FCD5 (x_out)

        #second theta
        x = self.FCD6 (x)
        x = F.leaky_relu (x)

        x = self.FCD7 (x)
        x = F.leaky_relu (x)

        x = self.FCD8 (x)
        x = F.leaky_relu (x)

        x = self.FCD9 (x)
        x = F.leaky_relu (x)

        x = self.FCD10 (x)
        x = F.leaky_relu (x)


        theta_2 = self.FCD11 (x)

        return theta_1, theta_2
 
    def forward(self, x, src_img = None, src_mask=None, use_src_mask=False):

        # encoding
        z, mu, log_var = self.encode (x)

        """
        Instead of Passing of exact mask in the decoder
        just pass a rectangle around region of interest
        """
        src_clones = src_mask.clone()
        src_mask_bboxes = UtilityFunctions.extract_bbox (src_clones.detach().cpu().numpy())
        batch_recon = torch.zeros_like (src_mask)
        batch_recon_img = torch.zeros_like (src_mask)

        
        
        for idx,bbox in enumerate (src_mask_bboxes):
            src_clones[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1

        theta_1, theta_2 = self.decode (z, src_clones)

        batch_size = theta_1.size(0)

        for idx in range (batch_size):

            bbox = src_mask_bboxes[idx]

            src_mask_patch = src_mask[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)
            src_img_patch = src_img[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)

            theta = (theta_1[idx].unsqueeze(0), theta_2[idx].unsqueeze(0))
            outsize = (src_mask.size()[2:], src_img_patch.size()[2:])
            
            reconstruction, grid_t = self.T.transform_data_two_theta (src_mask[idx].unsqueeze(0), \
                theta, outsize,bbox)
            reconstruction_img = self.T.interpolate (src_img[idx].unsqueeze(0), grid_t, src_mask.size()[2:])

            batch_recon[idx] = reconstruction[0]
            batch_recon_img[idx] = reconstruction_img[0]


        # if not use_src_mask:    
        #     # decoding
        #     theta_1, theta_2 = self.decode (z, src_clones)        
        # else:
        #     theta_1, theta_2 = self.decode (z, src_mask)
            
        # reconstruction, velocities = self.T.transform_data(src_mask, theta_1, outsize=src_mask.size()[2:], return_velocities=True)
        # reconstruction_img = self.T.transform_data(src_img, theta_1, outsize=src_img.size()[2:], return_velocities=False)
        

        # ###Do patch wise transformation here
        # for idx,bbox in enumerate (src_mask_bboxes):
        #     src_mask_patch = src_mask[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)
        #     src_img_patch = src_img[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]].unsqueeze(0).unsqueeze(0)


        #     src_mask_patch_trsfmd, velocities_theta_2 = self.T.transform_data(src_mask_patch, theta_2[idx].unsqueeze(0), \
        #         outsize = src_mask_patch.size()[2:], return_velocities=True )

        #     if idx == 0:
        #         velocities_theta_2_norm = torch.norm (velocities_theta_2)
        #     else:
        #         velocities_theta_2_norm = velocities_theta_2_norm + torch.norm (velocities_theta_2_norm)

        #     src_img_patch_trsfmd = self.T.transform_data(src_img_patch, theta_2[idx].unsqueeze(0),\
        #         outsize = src_img_patch.size()[2:])

        #     #put them back into whole image
        #     reconstruction[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = src_mask_patch_trsfmd
        #     reconstruction_img[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = src_img_patch_trsfmd

        
        return batch_recon, mu, log_var, z, batch_recon_img