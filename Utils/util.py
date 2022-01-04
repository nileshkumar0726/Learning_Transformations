import numpy as np
from skimage.measure import regionprops
import os
import cv2
import torch
import re

from Constants import IMG_FOLDER

class UtilityFunctions:

    def __init__(self) -> None:
        pass


    @staticmethod
    def make_label_lookup(y_train):
        # make a dictionary of lists that, for each class,
        # contains indices of datapoints from that class in the training set
        classes = np.unique(y_train)
        class_indices = {}
        for label in classes:
            class_indices[label] = np.where(y_train == label)[0]
        return class_indices

    @staticmethod
    def make_pairs_list_KNN (X_train, y_train, neighbours_to_keep = 5):

        '''
        Args:
            X_train: Training data of shape [Batch_Size, Heigh, Width, Channels]
            y_train: Labels corresponding to X_train 
            neighbours_to_keep: Pair each image to neighbours_to_keep closest neighbours
        '''

        class_lookup = UtilityFunctions.make_label_lookup(y_train)
        classes = np.unique(y_train)
        transformation_pairs = {}
        transformation_pairs_class = [] 

        for label in classes:
            datapoints_class = class_lookup[label]
            for ts1 in datapoints_class:

                #print (X_train.shape)
                current_img = np.expand_dims (X_train[ts1] , axis = 0)

                if len(current_img.shape) == 4:
                    dists = np.linalg.norm((current_img.reshape(current_img.shape[0] , -1)\
                    - X_train[datapoints_class].reshape(X_train[datapoints_class].shape[0], -1)), axis = (1)) #check this for 3 channel images
                else:
                    dists = np.linalg.norm((current_img - X_train[datapoints_class]), axis = (1,2))
                idx = np.argsort(dists, axis = 0)[1:neighbours_to_keep+1] 
                #middle = len(dists)//2
                #idx = np.argsort(dists, axis = 0)[middle:middle+neighbours_to_keep] 

                for ts2 in datapoints_class[idx]:
                    transformation_pairs_class.append ((ts1, ts2))
            transformation_pairs[label] = np.array(transformation_pairs_class) 
            transformation_pairs_class = []   
        return transformation_pairs


    @staticmethod
    def extract_bbox (mask_images):
        """
        Given batch of images 
        extract bboxes for masks in each imge
        """
        mask_images = mask_images.squeeze()
        mask_images = mask_images.astype(np.int32)
        bboxes = []
        for mask_image in mask_images:
            #mask_images = mask_images.squeeze()
            #print (mask_image.shape)
            label_properties = regionprops(mask_image)
            bboxes.append (label_properties[0]['bbox']) ##(min_row, min_col) , (max_row, max_col)
        bboxes = np.array (bboxes)
        return bboxes



    @staticmethod
    def load_samples (start = 0, end = 200, size=(200,200)):

        data = []
        labels = None

        train_label_folder = IMG_FOLDER
        filenames = os.listdir (train_label_folder)
        complete_paths = []

        #filenames.sort ()

        for i in range (start, end):
            filename = filenames[i]
            complete_path = os.path.join(train_label_folder, filename)        
            img = cv2.imread (complete_path, 0)
            img = cv2.resize (img, size)
            if img.max() != 0:
                data.append (img)
                complete_paths.append (complete_path)
            
        labels = np.zeros((len(data), 1))
        data = np.array(data)
        
        return data/255.0, labels, complete_paths


    @staticmethod
    def load_sample_volume_labels (start = 0, end = 200, size=(200,200)):

        """
        Assign labels based on location in the volume 
        """

        data = []
        labels = []

        train_label_folder = IMG_FOLDER
        filenames = os.listdir (train_label_folder)

        #filenames.sort ()

        for i in range (start, end):
            filename = filenames[i]
            complete_path = os.path.join(train_label_folder, filename)        
            img = cv2.imread (complete_path, 0)
            img = cv2.resize (img, size)
            if img.max() != 0:
                data.append (img)
                label = filename.split('.')[0][-3:]
                label = int (re.sub("[^0-9]", "", label))
                labels.append (label)
            
        
        data = np.array(data)
        labels = np.array(labels)
        
        return data/255.0, labels



    @staticmethod
    def final_loss(bce_loss, mu, logvar, beta=0.1):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE , beta*KLD #scale KLD loss

