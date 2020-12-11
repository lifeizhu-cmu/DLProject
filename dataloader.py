import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from os import listdir
from os.path import isfile, join

def load_data():
    # train data path
    image_dir_l = '/content/camvid/CamVid/train_labels/'
    image_dir = '/content/camvid/CamVid/train/'

    images = [(image_dir+f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    masks = [(image_dir_l+f) for f in listdir(image_dir_l) if isfile(join(image_dir_l, f))]

    df_train = pd.DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])
    df1 = df_train.sort_values(by='images')['images'].reset_index()
    df2 = df_train.sort_values(by='masks')['masks'].reset_index()
    df_train['images'] = df1['images']
    df_train['masks'] = df2['masks']

    # val data path
    image_dir_l = '/content/camvid/CamVid/val_labels/'
    image_dir = '/content/camvid/CamVid/val/'

    images = [(image_dir+f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    masks = [(image_dir_l+f) for f in listdir(image_dir_l) if isfile(join(image_dir_l, f))]

    df_val = pd.DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])
    df1 = df_val.sort_values(by='images')['images'].reset_index()
    df2 = df_val.sort_values(by='masks')['masks'].reset_index()
    df_val['images'] = df1['images']
    df_val['masks'] = df2['masks']

    # test data path
    image_dir_l = '/content/camvid/CamVid/test_labels/'
    image_dir = '/content/camvid/CamVid/test/'

    images = [(image_dir+f) for f in listdir(image_dir) if isfile(join(image_dir, f))]
    masks = [(image_dir_l+f) for f in listdir(image_dir_l) if isfile(join(image_dir_l, f))]

    df_test = pd.DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])
    df1 = df_test.sort_values(by='images')['images'].reset_index()
    df2 = df_test.sort_values(by='masks')['masks'].reset_index()
    df_test['images'] = df1['images']
    df_test['masks'] = df2['masks']

    return df_train, df_val, df_test


def form_class_map(all_class=False):
    class_map_df = pd.read_csv('/content/camvid/CamVid/class_dict.csv')
    class_map = []
    class_index = []
    class_name = []

    all_class = False

    for index, item in class_map_df.iterrows():
        class_map.append(np.array([item['b'], item['g'], item['r']]))
        if all_class or item['class_11'] == 1:
            class_index.append(index)
            class_name.append(item['name'])

    return class_map, class_index, class_name


def form_2D_label(mask, class_map):
    mask = mask.astype(np.uint8)
    label = np.zeros(mask.shape[:2], dtype= np.uint8)
    
    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i
    
    return label


class ImageDataset(Dataset):
    def __init__(self, df):
        self.image_list = df["images"].values.tolist()
        self.mask_list = df["masks"].values.tolist()
        self.class_map ,_ ,_ = form_class_map()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        resized_shape = (256, 256)
        img = cv2.imread(self.image_list[index])
        mask = cv2.imread(self.mask_list[index])
        img = cv2.resize(img,(resized_shape[1], resized_shape[0])).transpose(2,0,1)/255.
        mask = cv2.resize(mask,(resized_shape[1], resized_shape[0])).astype(np.uint8)
        label = form_2D_label(mask, self.class_map)
        return torch.Tensor(img), torch.Tensor(label)