import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tqdm
from tqdm.auto import tqdm as tqdmp
import torch
from torchvision import transforms
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
        self.path = path

    def load_df(self, csvpath, imagepath):
        self.csvpath = ('/').join([self.path, csvpath])
        self.df = pd.read_csv(self.csvpath)
        return self.df

    def generate_image_path(self):
        self.train_df = self.load_df('train.csv', 'train_images')
        self.test_df = self.load_df('test.csv', 'test_images')

        print(f"train: {self.train_df.shape}  test: {self.test_df.shape}")
        print(f"unique labels: {self.train_df.label_group.nunique()}")

        self.dataLoader.preprocess_train(self.train_df)
        self.dataLoader.preprocess_test(self.test_df)

    def preprocess_train(self, df):
        df['path'] = self.path + '/train_images/' + df['image']
        df.to_csv(self.path + '/train_proc.csv')

    def preprocess_test(self, df):
        df['path'] = self.path + '/test_images/' + df['image']
        df.to_csv(self.path + '/test_proc.csv')


class ImageHandler:
    def image_shape(self, image_path):
        im = cv2.imread(image_path)
        return str(im.shape)

    def standardize_image(self, width, height, orig_path, new_path):
        torch.manual_seed(17)
        self.image_viz(orig_path)
        image = cv2.imread(orig_path)
        dim = (width, height)
        # resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        tran = transforms.ToTensor()  # Convert the numpy array (C, H, W) Tensor format and /255 normalize to [0, 1.0]
        img_tensor = tran(resized) # (C,H,W), channel order (B, G, R)
        torch.save(img_tensor, new_path)

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

    # Generate shape of image, resize them to 200 x 200, as that is the min found in this set
    # Then generate the array which represents each image
    # Divide them all by 255 to scale them
    # Persist these images in disk before next steps


class MatchFinder:
    def match_matrix(self, phash_array):
        """ A function that checks for matches by phash value,
        takes phash values as input, outputs diff matrix as a df """
        phashs = phash_array.apply(lambda x: imagehash.hex_to_hash(x))
        phash_matrix = pd.DataFrame()
        for idx, i in enumerate(phash_array):
            phash_matrix = pd.concat([phash_matrix, phashs - imagehash.hex_to_hash(i)], axis=1)
        phash_matrix.columns = range(len(phash_array))
        return phash_matrix


class GenerateDataset:

    def generate_matching_pairs(self, path, matches):
        matchingsets = []
        for i in range(len(matches)):
            matchingsets.append(matches.iloc[i, :][matches.iloc[i, :] == 0].index.values)
        matchingsets = pd.Series(matchingsets)
        pairs = matchingsets[matchingsets.apply(lambda x: len(x) == 2)]
        triplets = matchingsets[matchingsets.apply(lambda x: len(x) == 3)]
        quartets = matchingsets[matchingsets.apply(lambda x: len(x) == 4)]

        matching_pairs = []
        for idx, val in pairs.items():
            matching_pairs.append(val)

        for idx, value in triplets.items():
            matching_pairs.append([value[0], value[1]])
            matching_pairs.append([value[1], value[2]])
            matching_pairs.append([value[0], value[2]])

        for idx, value in quartets.items():
            matching_pairs.append([value[0], value[1]])
            matching_pairs.append([value[1], value[2]])
            matching_pairs.append([value[0], value[2]])
            matching_pairs.append([value[0], value[3]])
            matching_pairs.append([value[1], value[3]])
            matching_pairs.append([value[2], value[3]])

        final = list(set([tuple(t) for t in matching_pairs]))

        matching = pd.DataFrame().from_records(final, columns=['one', 'two'])
        matching = matching.sort_values(by=['one'])
        matching.to_csv(path + '/matching_pairs.csv')

        non_matching_pairs = []
        counter = 0
        for j in range(0, len(matching_pairs)):
            if j not in matching['one']:
                #doesn't work, manually edit if there is a subsequent image pair match
                #non_matching_pairs.append([j, j+ 1])
                counter += 1
            if counter == matching.shape[0]:
                break
        non_matching = pd.DataFrame().from_records(non_matching_pairs, columns=['one', 'two'])
        non_matching = non_matching.sort_values(by=['one'])
        non_matching.to_csv(path + '/non_matching_pairs.csv')
        return

    def merged_dataset(self, index_df, train_df):
        # Merge data from training set with image pairs [posting_id, image, image_phash, title, label_group]
        # from train_df
        train_df['index'] = train_df.index

        dataset_one = train_df.loc[index_df['one']]
        dataset_one['idx'] = index_df.index
        dataset_two = train_df.loc[index_df['two']]
        dataset_two['idx'] = index_df.index

        dataset = pd.merge(dataset_one, dataset_two, on='idx', suffixes=['_1', '_2'])
        dataset = pd.merge(dataset, index_df, left_index=True, right_index=True)
        return dataset


