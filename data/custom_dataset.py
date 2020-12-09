from data.base_dataset import BaseDataset
import numpy as np


class CustomDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.root = opt.dataroot
        data = np.load(opt.dataroot)
        self.X = data['arr_0']
        self.Y = data['arr_1']

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
        return {'A': np.rollaxis(self.X[index], axis=-1), 'B': np.rollaxis(self.Y[index], axis=-1), 'A_paths': self.root, 'B_paths': self.root}

    def __len__(self):
        """ Return the total number of images in the dataset. """
        return self.X.shape[0]
