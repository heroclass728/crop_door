import glob
import os
import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
from settings import ROOT_DIR


class TrainData:

    def __init__(self):

        self.train_img = os.path.join(ROOT_DIR, 'training_data')

    def prepare_train_data(self):

        door_images_path = glob.glob(os.path.join(self.train_img, '*.*'))

        self.create_feature_matrix(frame_path=door_images_path)

    def create_features(self, img):

        # flatten three channel color image
        color_features = img.flatten()
        # convert image to greyscale
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get HOG features from greyscale image
        hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
        # combine color and hog features into a single array
        flat_features = np.hstack(color_features)

        return flat_features

    def create_feature_label_matrix(self, frame_path):

        features_list = []
        label_list = []

        for fm_path in frame_path:
            # load image
            img_dir = os.path.basename(fm_path)
            img = cv2.imread(fm_path)
            # get features for image
            image_features = self.create_features(img=img)
            features_list.append(image_features)

        # convert list of arrays into a matrix
        feature_matrix = np.array(features_list)

        return feature_matrix

    print('Feature matrix shape is: ', feature_matrix.shape)

    # define standard scaler
    ss = StandardScaler()
    # run this on our feature matrix
    bees_stand = ss.fit_transform(feature_matrix)

    pca = PCA(n_components=500)
    # use fit_transform to run PCA on our standardized matrix
    bees_pca = ss.fit_transform(bees_stand)
    # look at new shape
    print('PCA matrix shape is: ', bees_pca.shape)
