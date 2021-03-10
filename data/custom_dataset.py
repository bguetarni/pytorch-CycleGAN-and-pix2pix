import random
from data.base_dataset import BaseDataset
import numpy as np
import os
import cv2
import torch


class CustomDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.mean = [0.3258, 0.3761, 0.3852]  # BGR values
        self.std = [0.0280, 0.0275, 0.0372]  # BGR values
        self.n_blurred = opt.input_nc//3  # number of blurred images to use
        self.n_pano = opt.output_nc//3  # number of parts to divide the pano into
        self.root = opt.dataroot
        if not isinstance(self.root, str):
            self.root = str(self.root)
        self.root = self.root if self.root[-1] == '/' else (self.root[-1] + '/')
        samples = os.listdir(self.root + 'inputs/')  # list of inputs samples
        random.shuffle(samples)  # randomize the order
        nb_samples = len(samples)
        self.X = np.empty((nb_samples, 96, 128, 3*self.n_blurred), dtype='uint8')
        self.Y = np.empty((nb_samples, 96, 128, 3*self.n_pano), dtype='uint8')
        for i, s in enumerate(samples):
            print('\rLoading training set {}/{}     '.format(i+1, len(samples)), end='')
            # input
            x = None
            images_list = os.listdir(self.root + 'inputs/' + s)
            images_list.sort()
            for img in images_list:
                # no need to convert to RGB since pix2pix process BGR images
                img = cv2.imread(self.root + 'inputs/' + s + '/' + img)
                img = cv2.resize(img, (128, 96))
                if x is None:
                    x = img
                else:
                    x = np.concatenate((x, img), axis=-1)
            self.X[i] = x

            # output
            y = cv2.imread(self.root + 'outputs/{}.jpg'.format(s))
            h, w = y.shape[0], y.shape[1]
            new_w = round(w/(h/96))  # computes the weight to resize the pano to
            y = cv2.resize(y, (new_w, 96))
            diff = self.n_pano*128 - y.shape[1]
            tmp = np.ones((96, 128*self.n_pano, 3), dtype='uint8')*255
            # fill the pixels with the pano so there is as much whit pixels at left as at right
            tmp[:, round(diff/2):round(diff/2) + y.shape[1], :] = y
            y = tmp
            for j in range(self.n_pano):
                self.Y[i, :, :, 3*j:3*(j+1)] = y[:, 128*j:128*(j+1), :]

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
        A = (self.X[index].astype('float32')/255. - self.mean*self.n_blurred)/(self.std*self.n_blurred)
        B = (self.Y[index].astype('float32')/255.)*2 - 1  # last layer of pix2pix is Tanh -> [-1, +1]

        # convert to Tensor
        A = torch.Tensor(A)
        B = torch.Tensor(B)

        # move channels dimension
        A = torch.movedim(A, 2, 0)
        B = torch.movedim(B, 2, 0)
        return {'A': A, 'B': B, 'A_paths': self.root + 'inputs/', 'B_paths': self.root + 'outputs/'}

    def __len__(self):
        """ Return the total number of images in the dataset. """
        return self.X.shape[0]
