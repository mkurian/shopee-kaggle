import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
from tqdm.auto import tqdm as tqdmp

tqdmp.pandas()

# Work with phash
import imagehash

import cv2, os
import skimage.io as io
from PIL import Image

# ignoring warnings
import warnings

warnings.simplefilter("ignore")


class DataLoader:
    def __init__(self, path):
        self.path = pd.read_csv(path)

    def load_train_df(self, csvpath, imagepath):
        self.train_csvpath = ('/').join(self.path, csvpath)
        self.train_df = pd.read_csv(self.train_csvpath)
        return self.train_df

class ImageHandler:

    # def __init__(self):

    def image_viz(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')

    def plot_image_pairs(self, df, index1, index2):
        for idx, path in enumerate([df.loc[index1, 'path'], df.loc[index2, 'path']]):
            plt.subplot(1, 2, idx + 1)
            self.image_viz(path)
        plt.show()

    def plot_image(self, df, index):
        for idx, path in enumerate([df.loc[index, 'path']]):
            plt.subplot(1, 2, idx + 1)
            self.image_viz(path)
        plt.show()