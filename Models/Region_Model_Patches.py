import torch
import torch.nn as nn
import torch.nn.functional as F
from libcpab import Cpab, CpabAligner




"""
Instead of passing complete images to encoder
just pass patches contatining tumor mask to 
the encoder


Todo: Changie naming convention to make it general
"""

class Region_Specific_Model_Liver_Patches_VAE_DI(nn.Module):
    def __init__(self, img_channels = 2, z_dim = 35):
        super(Region_Specific_Model_Liver_Patches_VAE_DI, self).__init__()
 
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


        self.bn1 = nn.BatchNorm2d (64)
        self.bn2 = nn.BatchNorm2d (64)
        self.bn3 = nn.BatchNorm2d (32)
        self.bn4 = nn.BatchNorm2d (32)
        # self.bn5 = nn.BatchNorm2d (16)
        # self.bn6 = nn.BatchNorm2d (16)
        # self.bn7 = nn.BatchNorm2d (8)
        # self.bn8 = nn.BatchNorm2d (8)
        # self.bn9 = nn.BatchNorm2d (8)
        # self.bn10 = nn.BatchNorm2d (8)
        # self.bn11 = nn.BatchNorm2d (8)

        self.enc1 = nn.Linear(in_features=13*26*32, out_features=1024)
        self.enc2 = nn.Linear (in_features=1024, out_features = 512 )
        self.enc3 = nn.Linear (in_features=512, out_features = 256 )
        self.enc4 = nn.Linear (in_features=256, out_features = 128 )
        self.enc5 = nn.Linear (in_features=128, out_features = z_dim ) #mu 
        self.enc6 = nn.Linear (in_features=128, out_features = z_dim ) #var

        self.bn12 = nn.BatchNorm1d(1024)
        self.bn13 = nn.BatchNorm1d(512)
        self.bn14 = nn.BatchNorm1d(256)
        self.bn15 = nn.BatchNorm1d(128)

        # decoder 
        self.dec1 = nn.Linear(in_features=z_dim + (36*36*8), out_features=8000)
        self.dec2 = nn.Linear(in_features=8000, out_features=8000)
        self.dec3 = nn.Linear(in_features=8000, out_features=4000)
        self.dec4 = nn.Linear(in_features=4000, out_features=4000)
        self.dec5 = nn.Linear(in_features=4000, out_features=2000)
        self.dec6 = nn.Linear(in_features=2000, out_features=2000)
        self.dec7 = nn.Linear (in_features=2000, out_features = 1000)
        self.dec8 = nn.Linear (in_features=1000, out_features = 1000)
        self.dec9 = nn.Linear(in_features=1000, out_features=574)

        self.bn16 = nn.BatchNorm1d(8000)
        self.bn17 = nn.BatchNorm1d(8000)
        self.bn18 = nn.BatchNorm1d(4000)
        self.bn19 = nn.BatchNorm1d(4000)
        self.bn20 = nn.BatchNorm1d(2000)
        self.bn21 = nn.BatchNorm1d(2000)
        self.bn22 = nn.BatchNorm1d(1000)
        self.bn23 = nn.BatchNorm1d(1000)

        
        self.convD1 = nn.Conv2d(1, out_channels = 64, kernel_size = 5, dilation=5) #output size 236
        self.convD2 = nn.Conv2d(64, out_channels = 64, kernel_size = 5, dilation=5) #output size 216
        self.convD3 = nn.Conv2d(64, out_channels = 32, kernel_size = 5, dilation=5) #output size 196
        self.convD4 = nn.Conv2d(32, out_channels = 32, kernel_size = 5, dilation=5) #output size 176
        self.convD5 = nn.Conv2d(32, out_channels = 16, kernel_size = 5, dilation=5) #output size 156
        self.convD6 = nn.Conv2d(16, out_channels = 16, kernel_size = 5, dilation=5) #output size 136
        self.convD7 = nn.Conv2d(16, out_channels = 8, kernel_size = 5, dilation=5) #output size 116
        self.convD8 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 96
        self.convD9 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 76
        self.convD10 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 56
        self.convD11 = nn.Conv2d(8, out_channels = 8, kernel_size = 5, dilation=5) #output size 36


        self.bnD1 = nn.BatchNorm2d (64)
        self.bnD2 = nn.BatchNorm2d (64)
        self.bnD3 = nn.BatchNorm2d (32)
        self.bnD4 = nn.BatchNorm2d (32)
        self.bnD5 = nn.BatchNorm2d (16)
        self.bnD6 = nn.BatchNorm2d (16)
        self.bnD7 = nn.BatchNorm2d (8)
        self.bnD8 = nn.BatchNorm2d (8) 
        self.bnD9 = nn.BatchNorm2d (8)
        self.bnD10 = nn.BatchNorm2d (8)
        self.bnD11 = nn.BatchNorm2d (8)

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
        x_out = self.bn1(x_out)
        x_out = F.leaky_relu(x_out)

        x_out = self.conv2 (x_out)
        x_out = self.bn2(x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv3 (x_out)
        x_out = self.bn3(x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.conv4 (x_out)
        x_out = self.bn4(x_out)
        x_out = F.leaky_relu (x_out)

        # x_out = self.conv5 (x_out)
        # x_out = self.bn5(x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv6 (x_out)
        # x_out = self.bn6(x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv7 (x_out)
        # x_out = self.bn7(x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv8 (x_out)
        # x_out = self.bn8(x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv9 (x_out)
        # x_out = self.bn9(x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv10 (x_out)
        # x_out = self.bn10 (x_out)
        # x_out = F.leaky_relu (x_out)

        # x_out = self.conv11 (x_out)
        # x_out = self.bn11 (x_out)
        # x_out = F.leaky_relu (x_out)


        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.enc1 (x_out)
        x_out = self.bn12(x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc2 (x_out)
        x_out = self.bn13(x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.enc3 (x_out)
        x_out = self.bn14(x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.enc4 (x_out)
        x_out = self.bn15(x_out)
        x_out = F.leaky_relu (x_out)
        
        mu = self.enc5 (x_out)
        log_var = self.enc6 (x_out)

        z = self.reparameterize (mu, log_var)
        return z, mu, log_var

    def decode (self, x, src_img):

        src_img_out = self.convD1 (src_img)
        src_img_out = self.bnD1(src_img_out)
        src_img_out = F.leaky_relu(src_img_out)

        src_img_out = self.convD2 (src_img_out)
        src_img_out = self.bnD2(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD3 (src_img_out)
        src_img_out = self.bnD3(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD4 (src_img_out)
        src_img_out = self.bnD4(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD5 (src_img_out)
        src_img_out = self.bnD5(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD6 (src_img_out)
        src_img_out = self.bnD6(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD7 (src_img_out)
        src_img_out = self.bnD7(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD8 (src_img_out)
        src_img_out = self.bnD8(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD9 (src_img_out)
        src_img_out = self.bnD9(src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD10 (src_img_out)
        src_img_out = self.bnD10 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = self.convD11 (src_img_out)
        src_img_out = self.bnD11 (src_img_out)
        src_img_out = F.leaky_relu (src_img_out)

        src_img_out = src_img_out.view(src_img_out.size(0), -1)


        #decoder should spit out theta
        x = torch.cat((x,src_img_out), dim=1 )
        
        x_out = self.dec1 (x)
        x_out = self.bn16 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.dec2 (x_out)
        x_out = self.bn17 (x_out)
        x_out = F.leaky_relu (x_out)
        
        x_out = self.dec3 (x_out)
        x_out = self.bn18 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.dec4 (x_out)
        x_out = self.bn19 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.dec5 (x_out)
        x_out = self.bn20 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.dec6 (x_out)
        x_out = self.bn21 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.dec7 (x_out)
        x_out = self.bn22 (x_out)
        x_out = F.leaky_relu (x_out)

        x_out = self.dec8 (x_out)
        x_out = self.bn23 (x_out)
        x_out = F.leaky_relu (x_out)

        theta = self.dec9 (x_out)

        return theta
 
    def forward(self, x, src_mask=None, use_src_mask=False):

        # encoding
        z, mu, log_var = self.encode (x)



        if not use_src_mask:    
            # decoding
            theta = self.decode (z, x[:,0,:,:].unsqueeze(1))        
        else:
            theta = self.decode (z, src_mask)
            #theta = self.decode (z, x[:,0,:,:].unsqueeze(1))
            
        reconstruction, velocities = T.transform_data(src_mask, theta, outsize=img_dimensions, return_velocities=True)
        #reconstruction, velocities = T.transform_data(x[:,0,:,:].unsqueeze(1), theta, outsize=(66,86), return_velocities=True)
        return reconstruction, mu, log_var, z, velocities