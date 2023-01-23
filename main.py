#from train import train_vae
from train_vanilla import train_vae
from Utils.util import UtilityFunctions
from Constants import normalize, img_dimensions

if __name__ == "__main__":

#     train_imgs, train_labels, train_paths =\
#          UtilityFunctions.load_tumor_samples (start=50, end=140, normalize=normalize, size=img_dimensions)

#     checkpoint_path = "/home/stu3/s15/nk4856/Research/Learning_Transformations/logs/20221202-165025/model_vae_kits_tumor_fold_4_50_percent.pt"
#     #'model_vae_tumor.pt'

#     UtilityFunctions.generate_samples(train_imgs, checkpoint_path, 'Gen_samples')



    train_vae ()
