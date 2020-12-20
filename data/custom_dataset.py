import random
from data.base_dataset import BaseDataset
import numpy as np
import os
import cv2


class CustomDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        mean = [0.3258, 0.3761, 0.3852]  # BGR values
        std = [0.0280, 0.0275, 0.0372]  # BGR values
        n_pano = 2  # number of parts to divide the pano into
        n_blurred = 2  # number of blurred images to use
        self.root = opt.dataroot
        if not isinstance(self.root, str):
            self.root = str(self.root)
        self.root = self.root + '/' if self.root[-1] != '/' else self.root[-1]
        samples = os.listdir(self.root + 'inputs/')  # list of inputs samples
        random.shuffle(samples)  # randomize the order
        nb_samples = len(samples)
        X = np.empty((nb_samples, 96, 128, 3*n_blurred), dtype='float32')
        Y = np.ones((nb_samples, 96, 128, 3*n_pano), dtype='float32')
        for i, s in enumerate(samples):
            print('\rLoading training set {}/{}'.format(i+1, len(samples)), end='')
            # input
            x = None
            images_list = os.listdir(self.root + 'inputs/' + s)
            images_list.sort()
            images_list = images_list[:n_blurred]  # retrieve inly the wanted images
            for img in images_list:
                img = cv2.imread(os.listdir(self.root + 'inputs/' + s + '/' + img)).astype('float32')
                img = cv2.resize(img, (96, 128))
                # no need to convert to RGB since pix2pix process BGR images
                img = (img/255.0 - mean)/std
                if x is None:
                    x = img
                else:
                    x = np.concatenate((x, img), axis=-1)
            X[i] = x

            # output
            y = cv2.imread(self.root + 'outputs/{}.jpg'.format(s)).astype('float32')
            h, w = y.shape[0], y.shape[1]
            new_w = round(w/(h/96))  # computes the weight to resize the pano to
            y = cv2.resize(y, (new_w, 96))
            y = (y/255.0)*2 - 1  # last layer of pix2pix is Tanh -> [-1, +1]
            diff = n_pano*128 - y.shape[1]

            # fill the pixels with the pano so there is as much whit pixels at left as at right
            Y[:, round(diff/2):round(diff/2) + x.shape[1], :] = y
        # store the training set
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        return {'A': self.X[index], 'B': self.Y[index], 'A_paths': self.root + 'inputs/', 'B_paths': self.root + 'outputs/'}

    def __len__(self):
        """ Return the total number of images in the dataset. """
        return self.X.shape[0]
