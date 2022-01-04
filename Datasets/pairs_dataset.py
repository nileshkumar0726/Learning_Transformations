import torch
from torch.utils.data import Dataset
<<<<<<< HEAD
import cv2

class PairsDataset (Dataset):

    def __init__ (self, pairs, imgs, filenames):

        self.pairs = pairs[0]
        self.imgs = imgs
        self.filenames = filenames
=======

class PairsDataset (Dataset):

    def __init__ (self, pairs, imgs):

        self.pairs = pairs[0]
        self.imgs = imgs
>>>>>>> 335386a6a1bf5bdddd749f15a7a7796713330c2e

    def __getitem__ (self, idx):

        x_src_idx, x_tgt_idx = self.pairs[idx]
        x_src = self.imgs[x_src_idx]
        x_tgt = self.imgs[x_tgt_idx]

<<<<<<< HEAD
        x_src_img_path = self.filenames[x_src_idx].replace("Labels","Images").replace("label","image")
        x_src_img = cv2.imread (x_src_img_path, 0)/255.0

        x_src = torch.from_numpy (x_src).unsqueeze(0)
        x_tgt = torch.from_numpy (x_tgt).unsqueeze(0)
        x_src_img = torch.from_numpy(x_src_img).unsqueeze(0)

        return x_src, x_tgt, x_src_img 
=======
        x_src = torch.from_numpy (x_src).unsqueeze(0)
        x_tgt = torch.from_numpy (x_tgt).unsqueeze(0)

        return x_src, x_tgt 
>>>>>>> 335386a6a1bf5bdddd749f15a7a7796713330c2e

    def __len__ (self):

        return len(self.pairs)
        
    
