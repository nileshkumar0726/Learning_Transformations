import torch
from torch.utils.data import dataset

class PairsDataset (dataset):

    def __init__ (self, pairs, imgs):

        self.pairs = pairs
        self.imgs = imgs

    def __getitem__ (self, idx):

        x_src_idx, x_tgt_idx = self.pairs[idx]
        x_src = self.imgs[x_src_idx]
        x_tgt = self.imgs[x_tgt_idx]

        x_src = torch.from_numpy (x_src)
        x_tgt = torch.from_numpy (x_tgt)

        return x_src, x_tgt 

    def __len__ (self):

        return len(self.pairs)
        
    
