import numpy as np
from skimage.measure import regionprops
import os
import cv2
import torch
import re
from Models.Vanilla_VAE import Vanilla_VAE

from Constants import IMG_FOLDER, max_slice_no, TUMOR_SEPERATED_FOLDER,\
    img_dimensions, Checkpoint_folder, configuration, normalize, latent_dim, fold_3_50_percent_train


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
    def make_pairs_list_KNN (X_train, y_train, neighbours_to_keep = 8):

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
                idx = np.argsort(dists, axis = 0)[4:neighbours_to_keep+1] 
                #middle = len(dists)//2
                #idx = np.argsort(dists, axis = 0)[middle:middle+neighbours_to_keep] 

                for ts2 in datapoints_class[idx]:
                    transformation_pairs_class.append ((ts1, ts2))
            transformation_pairs[label] = np.array(transformation_pairs_class) 
            transformation_pairs_class = []   
        return transformation_pairs


    @staticmethod
    def make_pairs_list_modified_KNN (X_train, y_train, neighbours_to_keep = 10):

        '''
            Same as make_pairs_list_modified_KNN but with different distance function
            distance is based on tumor to tumor patch not entire images

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

                current_img = np.expand_dims (X_train[ts1] , axis = 0)
                dists = []
                for tgt_img in X_train[datapoints_class]:
                    
                    tgt_img = np.expand_dims (tgt_img, axis = 0)
                    # imgs = np.concatenate ((current_img, tgt_img) , axis = 0)
                    # bboxes = UtilityFunctions.extract_bbox (imgs)
                    # curr_bbox, tgt_bbox = bboxes[0], bboxes[1]
                    # curr_bbox, tgt_bbox = UtilityFunctions.match_bboxes (curr_bbox, tgt_bbox)

                    diff_matrix = current_img - tgt_img #UtilityFunctions.augmented_distance (current_img, tgt_img, curr_bbox, tgt_bbox)
                    dist = np.linalg.norm (diff_matrix.flatten())


                    dists.append (dist)

                dists = np.array (dists)
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
        
        if len(mask_images.shape) == 2: #in case there was just one mask 
            mask_images = np.expand_dims(mask_images, 0)

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
    def load_samples (start = 0, end = 200, size=(200,200), normalize=True):

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
            unique_counts = np.unique(img, return_counts = True)

            
            if img.max() != 0 and unique_counts[1][1] >= 70: #remove extra small objects
                
                data.append (img)
                complete_paths.append (complete_path)
            
        labels = np.zeros((len(data), 1))
        data = np.array(data)
        
        if normalize:
            return data/255.0, labels, complete_paths
        else:
            return data, labels, complete_paths



    @staticmethod
    def load_tumor_samples (start = 0, end = 200, size=(30,30), normalize=True):

        """
        For LiTs tumor dataset
        """

        data = []
        labels = None

        train_label_folder = TUMOR_SEPERATED_FOLDER
        actual_label_folder = IMG_FOLDER
        filenames = os.listdir (train_label_folder)
        complete_paths = []

        #filenames.sort ()

        for i in range (start, end):
            filename = filenames[i]
            
            case_no = filename[:5]
            if case_no not in fold_3_50_percent_train:
                continue
                
            complete_read_path = os.path.join(train_label_folder, filename)    
            img = cv2.imread (complete_read_path, 0)
            unique_counts = np.unique(img, return_counts = True)
            
            if img.max() != 0 and unique_counts[1][1] >= 80: #remove extra small tumors
                
                bbox = UtilityFunctions.extract_bbox(img)[0]
                img = img[bbox[0]-10:bbox[2]+10, bbox[1]-10:bbox[3]+10]
                img = cv2.resize (img, size, interpolation=cv2.INTER_NEAREST)
                data.append (img)
                complete_paths.append (complete_read_path)
            
        labels = np.zeros((len(data), 1))
        data = np.array(data)
        

        print ("DAta shape = ", data.shape)
        if normalize:
            return data/255.0, labels, complete_paths
        else:
            return data, labels, complete_paths


    @staticmethod
    def load_sample_volume_labels (start = 0, end = 200, size=(200,200)):

        """
        Assign labels based on location in the volume 
        """

        data = []
        labels = []

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
                _, slice_no = UtilityFunctions.extract_patient_slice (filename)
                slice_no = int (slice_no)
                
                if slice_no > 360:
                    slice_no = 360
                
                label = int(slice_no) // 36
                labels.append (label)
            
        
        data = np.array(data)
        labels = np.array(labels)
        
        return data/255.0, labels, complete_paths



    @staticmethod
    def final_loss(bce_loss, mu, logvar, beta=0.001):
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


    @staticmethod
    def extract_patient_slice (filename):
        
        """
        Given filename for an image
        this functionr returns the
        patient no. and slice no.
        """

        identifiers = re.sub("[^0-9]", "", filename)

        patient_no, slice_no = identifiers[:4], identifiers[4:] #we know patient no is always 4 in length

        return patient_no, slice_no


    @staticmethod 
    def union_bboxes (bbox_1, bbox_2):

        """
        Give two bboxes return a bbox that has union 
        of two bboxes
        """

        bbox = np.zeros_like (bbox_1)

        bbox[0] = min (bbox_1[0], bbox_2[0])
        bbox[1] = min (bbox_1[1], bbox_2[1])
        bbox[2] = max (bbox_1[2], bbox_2[2])
        bbox[3] = max (bbox_1[3], bbox_2[3])

        return bbox


    @staticmethod
    def augmented_distance (x1, x2, x1_bbox, x2_bbox):

        """
        calculated distance based on difference b/w
        tumors instead of whole images 

        Args: x1 & x2: numpy array of (H,W)
        bboxes used skimage regrionprops

        """

        src_roi = x1[:,x1_bbox[0]:x1_bbox[2], x1_bbox[1]:x1_bbox[3]]
        trgt_roi = x2[:,x2_bbox[0]:x2_bbox[2], x2_bbox[1]:x2_bbox[3]]
        
        diff = src_roi.squeeze() - trgt_roi.squeeze()

        return diff


    
    @staticmethod
    def match_bboxes (x_n_bbox, x_m_bbox):
        
        """
        This function matches the two bboxes 
        so they have equal height and width 
        for recon loss calculation 
        """

        src_roi_height = x_n_bbox[2] - x_n_bbox[0]
        src_roi_width = x_n_bbox[3] - x_n_bbox[1]

        trgt_roi_height = x_m_bbox[2] - x_m_bbox[0]
        trgt_roi_width = x_m_bbox[3] - x_m_bbox[1]

        #Handle Height difference, don't do anything if equal
        if src_roi_height < trgt_roi_height: #need to expand src roi
            height_diff = trgt_roi_height - src_roi_height
            half_height = int(height_diff/2)
            if  (x_n_bbox[0] - half_height) >= 0 and (x_n_bbox[2] + \
                (height_diff - half_height ) ) <= (img_dimensions[0]-1): #case where we can expand roi bottom and up equally
                x_n_bbox [0] -= half_height 
                x_n_bbox [2] += (height_diff - half_height)
            elif (x_n_bbox[0] - height_diff) >= 0:
                x_n_bbox[0] -= height_diff
            else:
                x_n_bbox[2] += height_diff
        
        elif trgt_roi_height < src_roi_height: #need to expand trgt roi
            height_diff = src_roi_height - trgt_roi_height
            half_height = int(height_diff/2)
            if  (x_m_bbox[0] - half_height) >= 0 and (x_m_bbox[2] + \
                (height_diff - half_height ) ) <= (img_dimensions[0] - 1): #case where we can expand roi bottom and up equally
                x_m_bbox [0] -= half_height 
                x_m_bbox [2] += (height_diff - half_height)
            elif (x_m_bbox[0] - height_diff) >= 0:
                x_m_bbox[0] -= height_diff
            else:
                x_m_bbox[2] += height_diff

        #Handle Width difference, don't do anything if same
        if src_roi_width < trgt_roi_width: #need to expand src roi
            width_diff = trgt_roi_width - src_roi_width
            half_width = int(width_diff/2)
            if  (x_n_bbox[1] - half_width) >= 0 and (x_n_bbox[3] + \
                (width_diff - half_width ) ) <= (img_dimensions[1] - 1): #case where we can expand roi bottom and up equally
                x_n_bbox [1] -= half_width 
                x_n_bbox [3] += (width_diff - half_width)
            elif (x_n_bbox[1] - width_diff) >= 0:
                x_n_bbox[1] -= width_diff
            else:
                x_n_bbox[3] += width_diff
        
        elif trgt_roi_width < src_roi_width: #need to expand trgt roi

            width_diff = src_roi_width - trgt_roi_width
            half_width = int(width_diff/2)
            if  (x_m_bbox[1] - half_width) >= 0 and\
                 (x_m_bbox[3] + (width_diff - half_width ) ) <= (img_dimensions[1] - 1): #case where we can expand roi bottom and up equally
                x_m_bbox [1] -= half_width 
                x_m_bbox [3] += (width_diff - half_width)
            elif (x_m_bbox[1] - width_diff) >= 0:
                x_m_bbox[1] -= width_diff
            else:
                x_m_bbox[3] += width_diff


        return x_n_bbox, x_m_bbox



    @staticmethod
    def save_checkpoint (epoch, model, optimizer, loss, checkpoint_dir):

        #PATH =  os.path.join (Checkpoint_folder, configuration, str(epoch))
        #if not os.path.isdir(PATH):
        #    os.mkdir (PATH)
        PATH = os.path.join (checkpoint_dir, 'model_vae_kits_tumor_fold_3_50_percent.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)



    @staticmethod
    def generate_samples (src_images, model_checkpoint, out_folder):

        no_of_trans_samples = 40

        src_images = torch.FloatTensor(src_images).cuda()
        
        model = Vanilla_VAE ().cuda()
        checkpoint = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        rand_sample = torch.randn((no_of_trans_samples, latent_dim)).cuda()
        thetas = model.decode(rand_sample)

        for idx, theta in enumerate (thetas):

            #batch and color dimension to the image
            curr_img = src_images[3].unsqueeze(0).unsqueeze(0)

            out_image = model.T.transform_data(curr_img, theta.unsqueeze(0), outsize=curr_img.size()[2:])

            out_image = out_image.detach().cpu().numpy().squeeze()

            src_img_path = os.path.join (out_folder, 'src_img.png')
            cv2.imwrite (src_img_path, src_images[3].detach().cpu().numpy().squeeze()*255)

            dest_img_path = os.path.join (out_folder, 'dest_img_'+str(idx)+ '.png')
            cv2.imwrite (dest_img_path, out_image*255)


        















        




