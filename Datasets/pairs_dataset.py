import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from Constants import img_dimensions, TUMOR_SEPERATED_FOLDER, IMG_FOLDER

class PairsDataset (Dataset):

    def __init__ (self, pairs, imgs, filenames, isTumor = False):

        self.isTumor = isTumor
        
        for idx, key in enumerate(pairs.keys()):
            if idx == 0:
                self.pairs = pairs[key]
            else:
                if len(pairs[key]) > 0:
                    self.pairs = np.concatenate((self.pairs, pairs[key]), axis = 0)

        self.imgs = imgs
        self.filenames = filenames

    def __getitem__ (self, idx):

        x_src_idx, x_tgt_idx = self.pairs[idx]
        x_src = self.imgs[x_src_idx]
        x_tgt = self.imgs[x_tgt_idx]

        if self.isTumor:
            x_src_img_path = self.filenames[x_src_idx][:-6] + self.filenames[x_src_idx][-4:]
            x_src_img_path = x_src_img_path.replace (TUMOR_SEPERATED_FOLDER, IMG_FOLDER)
            x_src_img_path = x_src_img_path.replace("segmentation","volume").replace("seg","ct")\
                .replace("png","jpg")

            #incase a single slice has double digit tumors
            if x_src_img_path[-5] == '_':
                x_src_img_path = x_src_img_path[:-5] + x_src_img_path[-4:]

        else:
            x_src_img_path = self.filenames[x_src_idx].replace("Labels","Images").replace("label","image")
        
        x_src_img = cv2.imread (x_src_img_path, 0)/255.0
        x_src_img = cv2.resize (x_src_img, img_dimensions)

        x_src = torch.from_numpy (x_src).unsqueeze(0)
        x_tgt = torch.from_numpy (x_tgt).unsqueeze(0)
        x_src_img = torch.from_numpy(x_src_img).unsqueeze(0)

        return x_src, x_tgt, x_src_img 

    def __len__ (self):

        return len(self.pairs)
        
    
