import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab import Cpab, CpabAligner
from ..Constants import tess_size



"""
In this version source image and target image are both 
passed to the encoder as they are, in other versions 
the only the region of interests may be passed from 
both images
Image Size (200x200)
"""
class Region_Specific_VAE(nn.Module):
    def __init__(self, img_channels = 2, z_dim = 35):
        super(Region_Specific_VAE, self).__init__()
 
        
        
        T = Cpab(tess_size=tess_size, backend='pytorch', device=device_T, zero_boundary=True)
        
        
        # encoder conv layers
        self.convE1 = nn.Conv2d(img_channels, out_channels = 64, kernel_size = 5, dilation=5) #output size (180)
        self.convE2 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size (160)
        self.convE3 = nn.Conv2d(32, out_channels = 16, kernel_size = 5, dilation=5) #output size (140)
        self.convE4 = nn.Conv2d(16, out_channels = 16, kernel_size = 5, dilation=5) #output size (120)
        self.convE5 = nn.Conv2d(16, out_channels = 8, kernel_size = 5, dilation=5) #output size (100)
        self.convE6 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size (80)
        self.convE7 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size (60)
        self.convE8 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size (40)
        self.convE9 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size (20)

        #Encode fully connected layers
        self.FCE1 = nn.Linear(in_features=20*20*8, out_features=1024)
        self.FCE2 = nn.Linear (in_features=1024, out_features = 512 )
        self.FCE3 = nn.Linear (in_features=512, out_features = 128 )
        self.FCE4 = nn.Linear (in_features=128, out_features = z_dim ) #mu 
        self.FCE5 = nn.Linear (in_features=128, out_features = z_dim ) #var

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


        # decoder full connected layers
        self.FCD1 = nn.Linear(in_features=z_dim + (20*20*8), out_features=2400)
        self.FCD2 = nn.Linear(in_features=2400, out_features=2000)
        self.FCD3 = nn.Linear (in_features=2000, out_features = 1800)
        self.FCD4 = nn.Linear (in_features=1800, out_features = 1700)
        self.FCD5 = nn.Linear(in_features=1700, out_features=1598)

        

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
        x_out = self.convE1 (x)
        x_out = F.leaky_relu(x_out)

        x_out = self.convE2 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE4 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE5 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE6 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE7 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE8 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.convE9 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.FCE1 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.FCE2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCE3 (x_out)
        x_out = F.leaky_relu (x_out)
        
        mu = self.FCE4 (x_out)
        log_var = self.FCE5 (x_out)

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

        #decoder should spit out theta
        x = torch.cat((x,src_img_out), dim=1 )
        
        x_out = self.FCD1 (x)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD2 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.FCD3 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.FCD4 (x_out)
        x_out = F.leaky_relu (x_out)

        theta = self.FCD5 (x_out)

        return theta
 
    def forward(self, x, src_mask=None, use_src_mask=False):

        # encoding
        z, mu, log_var = self.encode (x)

        """
        Instead of Passing of exact mask in the decoder
        just pass a rectangle around region of interest
        """
        src_clones = x[:,0,:,:].unsqueeze(1).clone()
        src_mask_bboxes = extract_bbox (src_clones.detach().cpu().numpy())
        
        for idx,bbox in enumerate (src_mask_bboxes):
            src_clones[idx,0,bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1

        if not use_src_mask:    
            # decoding
            theta = self.decode (z, src_clones)        
        else:
            theta = self.decode (z, src_mask.unsqueeze(1))
            
        reconstruction, velocities = T.transform_data(x[:,0,:,:].unsqueeze(1), theta, outsize=x[:,0,:,:].unsqueeze(1).size()[2:], return_velocities=True)
        return reconstruction, mu, log_var, z, velocities