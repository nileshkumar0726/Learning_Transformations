#from train import train_vae
from train_vanilla import train_vae
from Utils.util import UtilityFunctions
from Constants import normalize, img_dimensions

if __name__ == "__main__":

    train_imgs, train_labels, train_paths =\
         UtilityFunctions.load_tumor_samples (start=0, end=40, normalize=normalize, size=img_dimensions)

    checkpoint_path = 'model_vae_tumor.pt'

    UtilityFunctions.generate_samples(train_imgs, checkpoint_path, 'Gen_samples')



    #train_vae ()
