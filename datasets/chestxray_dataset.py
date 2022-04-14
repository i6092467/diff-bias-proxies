"""
Data loaders for MIMIC-CXR
"""
import torch

import numpy as np

import pandas as pd

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from utils.misc_utils import set_seeds

from sklearn.model_selection import train_test_split


class ChestXRay_mimic_DatasetGenerator(Dataset):

    def __init__(self, imgs, img_list, label_list, attribute_list, transform,
                 class_names=['Enlarged Cardiomediastinum', 'No Finding']):

        self.listImageLabels = []
        self.listImageAttributes = []
        self.transform = transform
        imgLabel_cnt = np.zeros(len(class_names))
        self.imgs = imgs

        # iterate over imgs
        for i in range(len(img_list)):

            imageLabel = label_list.iloc[i]
            imageAttr = attribute_list[i]

            if imageLabel[0] != 1:
                imgLabel = 0
                imgLabel_cnt = imgLabel_cnt + [0, 1]
            else:
                imgLabel = 1
                imgLabel_cnt = imgLabel_cnt + [1, 0]

            self.listImageLabels.append(imgLabel)
            self.listImageAttributes.append(imageAttr)
        print(imgLabel_cnt)

    def __getitem__(self, index):

        imageData = Image.fromarray(self.imgs[index]).convert('RGB')

        image_label = self.listImageLabels[index]

        image_attr = self.listImageAttributes[index]

        if self.transform != None: imageData = self.transform(imageData)

        return imageData, image_label, image_attr

    def __len__(self):

        return len(self.listImageLabels)


def train_test_split_ChestXray_mimic(root_dir, prot_attr='gender', priv_class='M', unpriv_class='F',
                                     train_prot_ratio=0.75, seed=42,
                                     class_names=['Enlarged Cardiomediastinum', 'No Finding']):
    img_mat = np.load(root_dir + 'files_128.npy')

    df = pd.read_csv(root_dir + 'meta_data.csv')
    cnt_dis = len(df[df[class_names[0]] == 1])
    df = pd.concat([df[df[class_names[0]] == 1], df[df[class_names[1]] == 1].sample(n=int(cnt_dis))])
    print('Number of images total: ', len(df))

    # patient id split
    patient_id = sorted(list(set(df['subject_id'])))
    train_val_split = 0.5
    test_val_split = 0.5
    train_idx, test_val_idx = train_test_split(patient_id, test_size=train_val_split, shuffle=True, random_state=seed)
    test_idx, val_idx = train_test_split(test_val_idx, test_size=test_val_split, shuffle=True, random_state=seed)
    print('Number of patients in the training set: ', len(train_idx))
    print('Number of patients in the val set: ', len(val_idx))
    print('Number of patients in the test set: ', len(test_idx))

    # get the train dataframe and sample
    df_train = df[df['subject_id'].isin(train_idx)]
    attr_ratio = train_prot_ratio / (1 - train_prot_ratio)
    cnt_attr = len(df_train[(df_train[prot_attr] == priv_class)])
    if cnt_attr / attr_ratio > len(df_train[(df_train[prot_attr] == unpriv_class)]):
        cnt_attr = len(df_train[(df_train[prot_attr] == unpriv_class)]) * attr_ratio
    df_train = pd.concat([df_train[(df_train[prot_attr] == priv_class)].sample(n=int(cnt_attr)),
                          df_train[(df_train[prot_attr] == unpriv_class)].sample(n=int(cnt_attr / attr_ratio))])
    df_train = df_train.sort_values(by=['subject_id'])
    print('Number of images in train set: ', len(df_train))

    train_list = sorted(df.index[df['dicom_id'].isin(df_train['dicom_id'])].tolist())
    train_label = df_train[class_names]
    train_attr = list(0 + (df_train[prot_attr] == priv_class))
    train_imgs = img_mat[train_list, :, :]
    print('Number of priveleged images in train set: ', sum(train_attr))

    # get the val dataframe and sample
    df_val = df[df['subject_id'].isin(val_idx)]
    cnt_attr = len(df_val[(df_val[prot_attr] == priv_class)])
    if cnt_attr > len(df_val[(df_val[prot_attr] == unpriv_class)]):
        cnt_attr = len(df_val[(df_val[prot_attr] == unpriv_class)])
    df_val = pd.concat([df_val[(df_val[prot_attr] == priv_class)].sample(n=int(cnt_attr)),
                        df_val[(df_val[prot_attr] == unpriv_class)].sample(n=int(cnt_attr))])
    df_val = df_val.sort_values(by=['subject_id'])
    print('Number of images in val set: ', len(df_val))

    val_list = sorted(df.index[df['dicom_id'].isin(df_val['dicom_id'])].tolist())
    val_label = df_val[class_names]
    val_attr = list(0 + (df_val[prot_attr] == priv_class))
    val_imgs = img_mat[val_list, :, :]
    print('Number of priveleged images in val set: ', sum(val_attr))

    # get the test dataframe and sample
    df_test = df[df['subject_id'].isin(test_idx)]
    cnt_attr = len(df_test[(df_test[prot_attr] == priv_class)])
    if cnt_attr > len(df_test[(df_test[prot_attr] == unpriv_class)]):
        cnt_attr = len(df_test[(df_test[prot_attr] == unpriv_class)])
    df_test = pd.concat([df_test[(df_test[prot_attr] == priv_class)].sample(n=int(cnt_attr)),
                         df_test[(df_test[prot_attr] == unpriv_class)].sample(n=int(cnt_attr))])
    df_test = df_test.sort_values(by=['subject_id'])
    print('Number of images in test set: ', len(df_test))

    test_list = sorted(df.index[df['dicom_id'].isin(df_test['dicom_id'])].tolist())
    test_label = df_test[class_names]
    test_attr = list(0 + (df_test[prot_attr] == priv_class))
    test_imgs = img_mat[test_list, :, :]
    print('Number of priveleged images in test set: ', sum(test_attr))

    return train_list, val_list, test_list, train_label, val_label, test_label, train_attr, val_attr, test_attr, \
           train_imgs, val_imgs, test_imgs


def get_ChestXRay_mimic_dataloaders(device, root_dir, prot_attr='gender', priv_class='M', unpriv_class='F', train_prot_ratio=0.75,
                                    class_names=['Enlarged Cardiomediastinum', 'No Finding'], batch_size=256,
                                    num_workers=1, seed=42):
    print('Using {} device'.format(device))

    set_seeds(seed)

    train_list, val_list, test_list, train_label, val_label, test_label, \
    train_attr, val_attr, test_attr, train_imgs, val_imgs, test_imgs = \
        train_test_split_ChestXray_mimic(root_dir=root_dir, prot_attr=prot_attr, priv_class=priv_class,
                                         unpriv_class=unpriv_class, train_prot_ratio=train_prot_ratio, seed=seed,
                                         class_names=class_names)

    # Transformations
    transResize = 224
    transformList = []
    transformList.append(transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=(5)))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.Resize(size=transResize))
    transformList.append(transforms.ToTensor())
    train_transform = transforms.Compose(transformList)

    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.ToTensor())
    test_transform = transforms.Compose(transformList)

    # Datasets
    image_datasets = {'train': ChestXRay_mimic_DatasetGenerator(imgs=train_imgs,
                                                                img_list=train_list,
                                                                label_list=train_label,
                                                                attribute_list=train_attr,
                                                                transform=train_transform,
                                                                class_names=class_names),
                      'val': ChestXRay_mimic_DatasetGenerator(imgs=val_imgs,
                                                              img_list=val_list,
                                                              label_list=val_label,
                                                              attribute_list=val_attr,
                                                              transform=test_transform,
                                                              class_names=class_names),
                      'test': ChestXRay_mimic_DatasetGenerator(imgs=test_imgs,
                                                               img_list=test_list,
                                                               label_list=test_label,
                                                               attribute_list=test_attr,
                                                               transform=test_transform,
                                                               class_names=class_names)}

    # Dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)

    # Data loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes
